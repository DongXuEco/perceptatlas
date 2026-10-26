options(stringsAsFactors = FALSE)

script_arg <- grep("^--file=", commandArgs(), value = TRUE)
script_path <- sub("^--file=", "", script_arg[1])
package_root <- normalizePath(file.path(dirname(script_path), "../.."), winslash = "/", mustWork = TRUE)
output_dir <- file.path(package_root, "outputs", "Extended_Data_Fig_3")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

city_csv <- file.path(package_root, "data/shared/Study_city_analysis.csv")
zone_csv <- file.path(package_root, "data/shared/Study_city_zone_analysis.csv")
predictor_csv <- file.path(package_root, "data/Extended_Data_Fig_3/predictors/Extended_Data_Fig_3_final_predictors_recommended.csv")

message("Reading final predictor set...")
pred <- data.table::fread(predictor_csv, encoding = "UTF-8")
predictors <- pred$column

skewness <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) < 3) return(NA_real_)
  s <- stats::sd(x)
  if (!is.finite(s) || s == 0) return(0)
  mean((x - mean(x))^3) / (s^3)
}

prepare_matrix <- function(data, cols) {
  cols <- cols[cols %in% names(data)]
  x <- as.data.frame(data[, ..cols])
  for (nm in names(x)) {
    v <- suppressWarnings(as.numeric(x[[nm]]))
    sk <- skewness(v)
    finite <- v[is.finite(v)]
    if (length(finite) > 0 && min(finite, na.rm = TRUE) >= 0 && is.finite(sk) && abs(sk) > 1) {
      v <- log1p(v)
    }
    med <- stats::median(v, na.rm = TRUE)
    if (!is.finite(med)) med <- 0
    v[!is.finite(v) | is.na(v)] <- med
    x[[nm]] <- v
  }
  keep <- vapply(x, function(v) stats::sd(v) > 0, logical(1))
  x <- x[, keep, drop = FALSE]
  as.data.frame(scale(x))
}

compute_exact_vif <- function(x) {
  cols <- names(x)
  rows <- vector("list", length(cols))
  n <- nrow(x)
  for (i in seq_along(cols)) {
    y <- x[[i]]
    others <- as.matrix(x[, -i, drop = FALSE])
    fit <- stats::lm.fit(x = cbind(Intercept = 1, others), y = y)
    rss <- sum(fit$residuals^2)
    tss <- sum((y - mean(y))^2)
    r2 <- if (tss > 0) 1 - rss / tss else NA_real_
    r2 <- max(min(r2, 1), 0, na.rm = TRUE)
    vif <- if (is.na(r2)) NA_real_ else if (r2 >= 0.999999999999) Inf else 1 / (1 - r2)
    rows[[i]] <- data.frame(
      column = cols[i],
      n = n,
      exact_vif = vif,
      r2_against_other_predictors = r2,
      stringsAsFactors = FALSE
    )
  }
  out <- do.call(rbind, rows)
  out[order(-out$exact_vif), ]
}

run_one <- function(csv, table_name) {
  message(paste0("Reading ", table_name, " table..."))
  df <- data.table::fread(csv, encoding = "UTF-8")
  x <- prepare_matrix(df, predictors)
  vif <- compute_exact_vif(x)
  data.table::fwrite(vif, file.path(output_dir, paste0("Extended_Data_Fig_3_vif_", table_name, ".csv")))
  vif
}

vif_city <- run_one(city_csv, "city")
vif_zone <- run_one(zone_csv, "city_zone")

