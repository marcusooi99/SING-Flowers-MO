# scrape_gbif.R
# -----------------------------------------------------------------------------
# Curate a small test set for the LM2 flower detector: herbarium specimen images
# of Southeast Asian Magnoliopsida (LM2's Plant Component Detector was trained on
# Magnoliopsida), pulled from GBIF via occ_search (no account needed).
#
#   1. Pull candidate specimens with images, per SEA country.
#   2. Dedupe, cap per species, sample down to TARGET.
#   3. Fetch image URLs for the sampled keys.
#   4. Download the images.
#   5. Write a labeling template (fill in flowering / has_bud / ... by eye).
#
# Held-out rationale: SEA specimens digitised by SEA/EU herbaria are almost
# certainly outside LM2's overwhelmingly North American training pull. Not proven,
# just plausible - good enough for a proof-of-concept.
# -----------------------------------------------------------------------------

library(rgbif)
library(dplyr)
library(stringr)
library(readr)
library(purrr)
library(fs)

set.seed(123)

# ---- config -----------------------------------------------------------------
SEA_COUNTRIES <- c("BN", "KH", "ID", "LA", "MY", "MM", "PH", "SG", "TH", "TL", "VN")
PER_COUNTRY   <- 100   # candidates pulled per country before sampling
TARGET        <- 500   # images to actually download
MAX_PER_SPP   <- 5     # cap so one common species can't dominate

IMG_DIR  <- "sample_images"
DATA_DIR <- "data"
dir_create(IMG_DIR)
dir_create(DATA_DIR)

`%||%` <- function(a, b) if (is.null(a) || length(a) == 0 || is.na(a[1])) b else a

mag_key <- name_backbone("Magnoliopsida")$usageKey
message("Magnoliopsida taxonKey = ", mag_key)

# ---- 1. candidate pool ----------------------------------------------------
pull_country <- function(ctry) {
  message("  ", ctry, " ...")
  occ_search(
    taxonKey       = mag_key,
    country        = ctry,
    basisOfRecord  = "PRESERVED_SPECIMEN",
    mediaType      = "StillImage",
    occurrenceStatus = "PRESENT",
    limit          = PER_COUNTRY
  )$data
}

pool <- map_dfr(SEA_COUNTRIES, pull_country) |>
  mutate(key = as.character(key)) |>
  filter(!is.na(species)) |>
  distinct(key, .keep_all = TRUE) |>
  distinct(institutionCode, catalogNumber, .keep_all = TRUE)

message(nrow(pool), " candidate specimens across ",
        n_distinct(pool$countryCode), " countries")

# ---- 2. cap per species + sample ---------------------------------------
capped <- pool |>
  group_by(species) |>
  slice_sample(n = MAX_PER_SPP) |>
  ungroup()

sampled <- slice_sample(capped, n = min(TARGET, nrow(capped)))
message(nrow(sampled), " sampled")

# ---- 3. fetch image URLs for the sampled keys ------------------------
# Passing a vector of gbifIds makes occ_search return a list keyed by id, each
# with that record's media. (Same pattern as the original script.)
media_res <- occ_search(gbifId = sampled$key, mediaType = "StillImage", limit = 500)

first_url <- function(id) {
  x <- media_res[[id]]
  if (is.null(x) || length(x$media) == 0) return(NA_character_)
  tryCatch(x$media[[1]][[id]][[1]]$identifier %||% NA_character_,
           error = function(e) NA_character_)
}

url_df <- tibble(
  key       = names(media_res),
  image_url = map_chr(names(media_res), first_url)
)

# ---- weak flowering hint (to eyeball, NOT ground truth) --------------
flower_rx <- regex("flower|flowering|\\bfl\\.|anthesis|in bloom|petal|infloresc",
                   ignore_case = TRUE)
tcols <- intersect(c("occurrenceRemarks", "fieldNotes", "reproductiveCondition"),
                   names(sampled))

sampled <- sampled |>
  left_join(url_df, by = "key") |>
  filter(!is.na(image_url)) |>
  distinct(image_url, .keep_all = TRUE) |>
  mutate(
    weak_flowering = if (length(tcols)) {
      str_detect(apply(across(all_of(tcols)), 1,
                       \(r) paste(na.omit(r), collapse = " ")), flower_rx)
    } else NA
  ) |>
  slice_sample(prop = 1) |>                       # shuffle
  mutate(image_id   = sprintf("HERB_%04d", row_number()),
         image_path = path(IMG_DIR, paste0(image_id, ".jpg"))) |>
  relocate(image_id, image_path)

message(nrow(sampled), " have a usable image URL")

# ---- 4. download images ---------------------------------------------
magic_ok <- function(p) {
  sig <- tryCatch(readBin(p, "raw", 4L), error = \(e) raw(0))
  length(sig) >= 3 && (
    identical(sig[1:3], as.raw(c(0xFF, 0xD8, 0xFF))) ||          # JPEG
    identical(sig[1:4], as.raw(c(0x89, 0x50, 0x4E, 0x47)))       # PNG
  )
}

get_img <- function(url, dest) {
  if (file_exists(dest) && magic_ok(dest)) return("cached")
  Sys.sleep(0.1)
  ok <- tryCatch({ download.file(url, dest, mode = "wb", quiet = TRUE); TRUE },
                 error = \(e) FALSE, warning = \(w) FALSE)
  if (!ok || !file_exists(dest))  return("error")
  if (file_size(dest) < 5000)     { file_delete(dest); return("too_small") }
  if (!magic_ok(dest))            { file_delete(dest); return("not_image") }
  "ok"
}

sampled$download_status <- map2_chr(sampled$image_url, sampled$image_path,
                                    get_img, .progress = TRUE)
print(count(sampled, download_status))

# ---- 5. outputs: candidate table + labeling template --------------
write_csv(sampled, path(DATA_DIR, "candidates.csv"))

sampled |>
  filter(download_status %in% c("ok", "cached")) |>
  transmute(
    image_id, image_path,
    scientificName, family, countryCode, year, gbif_key = key, image_url,
    weak_flowering,             # text hint only - confirm every one by eye
    flowering = NA_integer_,    # 1 = at least one OPEN flower visible
    has_bud   = NA_integer_,    # 1 = bud(s) present, no open flower
    has_fruit = NA_integer_,    # 1 = fruit / seed present
    usable    = NA_integer_,    # 0 = bad scan / label-only / multi-sheet
    notes     = NA_character_
  ) |>
  write_csv(path(DATA_DIR, "labels_template.csv"))

writeLines(c(
  sprintf("GBIF occ_search, accessed %s", Sys.Date()),
  sprintf("taxonKey = %s (Magnoliopsida)", mag_key),
  sprintf("country in {%s}", paste(SEA_COUNTRIES, collapse = ", ")),
  "basisOfRecord = PRESERVED_SPECIMEN, mediaType = StillImage, occurrenceStatus = PRESENT",
  sprintf("seed = 123 | candidates = %d | sampled = %d | downloaded = %d",
          nrow(pool), nrow(sampled),
          sum(sampled$download_status %in% c("ok", "cached")))
), path(DATA_DIR, "provenance.txt"))

message("Done. Fill in data/labels_template.csv by eye, save as data/labels.csv")
