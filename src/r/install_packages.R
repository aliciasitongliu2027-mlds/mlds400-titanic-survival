# This assignment uses only R's built-in packages.
# With r-base:4.4.3, stats and utils are already included, so no CRAN
# downloads are needed. This keeps the Docker build fast and reproducible.
required_packages <- c("stats", "utils")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0) {
  stop(paste("Missing required R packages:",
             paste(missing_packages, collapse = ", ")))
}
cat("R package check passed: stats and utils are available.\n")
