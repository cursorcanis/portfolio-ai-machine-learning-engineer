use anyhow::Result;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

/// Vector dimension for intfloat/multilingual-e5-small.
///
/// Same 384 dims as the BGE-small model this replaced, so the Qdrant
/// collection geometry is unchanged — but the vectors are NOT compatible.
/// Switching models requires re-indexing the whole collection.
pub const EMBED_DIM: u64 = 384;

/// E5 models are trained asymmetrically: stored documents and search queries
/// get different prefixes. Omitting them measurably degrades retrieval, and
/// fastembed does not add them for us — so they live here, applied by the
/// only two methods that touch the model.
const QUERY_PREFIX: &str = "query: ";
const PASSAGE_PREFIX: &str = "passage: ";

/// Thin wrapper around fastembed's TextEmbedding so the rest of the codebase
/// doesn't need to know which ONNX model we use.
///
/// Multilingual by design: `multilingual-e5-small` covers ~100 languages in a
/// shared vector space, so an English question retrieves Italian (or any
/// other language) source chunks without translating anything at ingest time.
pub struct Embedder {
    model: TextEmbedding,
}

impl Embedder {
    /// Load the multilingual E5-small model. On first call this downloads
    /// ~470 MB of ONNX weights into the fastembed cache directory.
    pub fn new() -> Result<Self> {
        let model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::MultilingualE5Small)
                .with_show_download_progress(true),
        )?;
        Ok(Self { model })
    }

    /// Embed a search query. Returns a 384-dim vector.
    ///
    /// Use this only for the text a user is searching *with*; stored content
    /// must go through [`Embedder::embed_batch`] so it gets the passage
    /// prefix instead.
    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let prefixed = format!("{QUERY_PREFIX}{text}");
        let mut out = self.model.embed(vec![prefixed], None)?;
        Ok(out.remove(0))
    }

    /// Embed a batch of documents destined for the vector store.
    pub fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let prefixed: Vec<String> = texts
            .into_iter()
            .map(|t| format!("{PASSAGE_PREFIX}{t}"))
            .collect();
        Ok(self.model.embed(prefixed, None)?)
    }
}
