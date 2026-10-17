use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use axum::routing::{delete, get, post};
use axum::Router;
use tower_http::cors::{Any, CorsLayer};
use tracing::info;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

use ai_advisor_rs::embeddings::Embedder;
use ai_advisor_rs::handler;
use ai_advisor_rs::knowledge::{advisor_names, load_knowledge_base};
use ai_advisor_rs::vector_store::VectorStore;
use ai_advisor_rs::AppState;

/// Resolve the canonical log file path for this backend run.
///
/// Default: `<cwd>/logs/server.log`. Override the directory with `LOG_DIR`
/// or the full path with `LOG_FILE`. Per Beads ai_advisor-egz — no more
/// scattered Claude task `.output` files vs stale `server.log` files
/// depending on how the binary was launched.
fn resolve_log_path() -> PathBuf {
    if let Ok(p) = std::env::var("LOG_FILE") {
        if !p.trim().is_empty() {
            return PathBuf::from(p);
        }
    }
    let dir = std::env::var("LOG_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("logs"));
    dir.join("server.log")
}

#[tokio::main]
async fn main() {
    let log_path = resolve_log_path();
    let log_dir = log_path
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let log_file_name = log_path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "server.log".to_string());
    std::fs::create_dir_all(&log_dir).ok();

    // Dual-writer subscriber: human-readable colored stdout + plain-text
    // append-only file at the canonical path. The `_log_guard` must outlive
    // main() — when the non-blocking writer's guard is dropped, queued log
    // lines are dropped on the floor, so we shadow it into the function
    // scope and let it live until process exit.
    let file_appender = tracing_appender::rolling::never(&log_dir, &log_file_name);
    let (file_writer, _log_guard) = tracing_appender::non_blocking(file_appender);

    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("ai_advisor_rs=info"));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(tracing_subscriber::fmt::layer().with_writer(std::io::stdout))
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(file_writer)
                .with_ansi(false),
        )
        .init();

    // Startup banner — explicit log destination so future tail-readers (us,
    // future agents, Beads workflows) never have to guess where logs went.
    let abs_log_path = std::fs::canonicalize(&log_dir)
        .map(|p| p.join(&log_file_name))
        .unwrap_or_else(|_| log_path.clone());
    info!("== AI Advisor backend starting ==");
    info!("Logs: {} (also stdout)", abs_log_path.display());

    // Load .env.local (same file the Next.js app uses)
    let env_path = PathBuf::from("..").join("ai-advisor").join("ai-advisor").join(".env.local");
    if env_path.exists() {
        dotenvy::from_path(&env_path).ok();
        info!("Loaded env from {}", env_path.display());
    } else {
        dotenvy::dotenv().ok();
    }

    let openrouter_api_key = std::env::var("OPENROUTER_API_KEY")
        .expect("OPENROUTER_API_KEY must be set (in .env.local or environment)");

    // Resolve the data directory (same location as the Next.js app's data/)
    let data_dir = PathBuf::from("..")
        .join("ai-advisor")
        .join("ai-advisor")
        .join("data");

    // Derive the expert list by reading the (still on-disk) KB JSON.
    // Retrieval now happens against Qdrant, but we still need the set of names
    // so the prompt tells Gemini which voices it can channel.
    info!("Loading expert list from {}", data_dir.display());
    let kb = load_knowledge_base(&data_dir);
    let experts = advisor_names(&kb);
    info!("Loaded {} experts: {:?}", experts.len(), experts);
    drop(kb);

    // Init the local embedding model (downloads on first run, ~130 MB).
    info!("Initializing embedding model (intfloat/multilingual-e5-small)...");
    let embedder =
        Arc::new(Embedder::new().expect("failed to initialize fastembed TextEmbedding"));
    info!("Embedding model ready");

    // Connect to Qdrant
    let qdrant_url =
        std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://127.0.0.1:6334".to_string());
    info!("Connecting to Qdrant at {qdrant_url}");
    let vector_store = Arc::new(
        VectorStore::connect(&qdrant_url).expect("failed to connect to Qdrant"),
    );
    vector_store
        .ensure_collection()
        .await
        .expect("failed to ensure Qdrant collection (is Qdrant running? start it with scripts/start-qdrant.ps1)");

    let initial_window = ai_advisor_rs::load_fetch_window();
    info!(
        "Restored fetch window from disk: {} entries within the last hour",
        initial_window.len()
    );

    let embed_concurrency: usize = std::env::var("EMBED_CONCURRENCY")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(2);
    info!("Chat embedding concurrency: {embed_concurrency}");

    let state = Arc::new(AppState {
        experts,
        data_dir,
        openrouter_api_key,
        http_client: reqwest::Client::new(),
        embedder,
        embed_permits: Arc::new(tokio::sync::Semaphore::new(embed_concurrency)),
        vector_store,
        jobs: Arc::new(Mutex::new(HashMap::new())),
        recent_fetches: Arc::new(Mutex::new(initial_window)),
    });

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/api/chat", post(handler::chat_handler))
        .route(
            "/api/creators",
            post(handler::start_creator_job).get(handler::list_creators),
        )
        .route("/api/creators/{job_id}", get(handler::get_creator_job))
        .route("/api/experts", get(handler::list_all_experts))
        .route("/api/experts/search", post(handler::search_expert))
        .route("/api/experts/confirm", post(handler::confirm_expert))
        .route("/api/experts/ingest-url", post(handler::ingest_url))
        .route(
            "/api/experts/ingest-document",
            post(handler::ingest_document),
        )
        .route("/api/experts/{name}", delete(handler::delete_expert))
        .route("/api/experts/{name}/export", get(handler::export_expert))
        .route("/api/experts/{name}/videos", get(handler::list_expert_videos))
        .route("/api/limits", get(handler::get_limits))
        .fallback(handler::handle_error)
        .layer(cors)
        .with_state(state);

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3002);
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    info!("AI Advisor Rust server listening on http://{addr}");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
