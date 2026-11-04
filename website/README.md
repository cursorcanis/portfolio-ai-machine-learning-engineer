# aleaportfolio.com

Source for **https://aleaportfolio.com**, the one-page portfolio with the AI Advisor and Georgia homelessness case studies.

Plain static HTML with no build step: `index.html` (inline CSS), the charts in `img/`, and the two one-page snapshot PDFs. All text and numbers are copied from the reviewed case-study documents.

## Hosting

- S3 bucket `aleaportfolio-frontend`, served by CloudFront (apex + `www`, HTTPS via ACM). DNS in Route 53.
- The top-menu "AI Advisor" link goes to `https://alfredoalea.org/?project=11`.

## Deploy

Requires an AWS CLI profile with S3 + CloudFront access (here: `deploy`).

```bash
aws s3 sync . s3://aleaportfolio-frontend --delete --exclude README.md --profile deploy
aws cloudfront create-invalidation --distribution-id E25X4TT6CY4C3 --paths "/*" --profile deploy
```
