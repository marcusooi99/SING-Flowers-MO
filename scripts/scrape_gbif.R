# scrape_gbif.R

library(rgbif)
library(tidyverse)

# check data availability by SEA country ----
sea_countries <- c(
  "BN", "KH", "ID", "LA", "MY", "MM",
  "PH", "SG", "TH", "TL", "VN"
)

# country_counts <- map_dfr(
#   sea_countries,
#   \(ctry) {
# 
#     res <- occ_search(
#       country = ctry,
#       kingdomKey = 6,                 # Plantae
#       basisOfRecord = "PRESERVED_SPECIMEN",
#       mediaType = "StillImage",
#       occurrenceStatus = "PRESENT",
#       limit = 0
#     )
# 
#     tibble(
#       country = ctry,
#       n_records = res$meta$count
#     )
#   }
# )

set.seed(123)

# retrieve candidate pool of 500 images per country
candidate <- map_dfr(
  sea_countries,
  \(ctry) {
    
    occ_search(
      country = ctry,
      kingdomKey = 6,
      basisOfRecord = "PRESERVED_SPECIMEN",
      mediaType = "StillImage",
      occurrenceStatus = "PRESENT",
      limit = 500
    )$data
  }
) %>% 
  filter(!is.na(species))

candidate_balanced <- candidate %>%
  group_by(country, species) %>%
  slice_sample(n = 5) %>%
  ungroup()

# Now sample 45 images per country
sampled <- candidate_balanced %>%
  group_by(country) %>%
  slice_sample(n = 45) %>%
  ungroup()

remaining <- candidate %>%
  anti_join(sampled, by = "key")

extra <- remaining %>%
  slice_sample(n = 5)

sampled <- bind_rows(sampled, extra)

nrow(sampled)

sampled %>%
  count(country)

sampled %>%
  count(country, species) %>%
  arrange(country, desc(n))

# retrieving media url ----
selected_res <- occ_search(
  gbifId = sampled$key,
  mediaType = "StillImage",
  limit = 500
)

image_urls <- sapply(
  names(selected_res),
  \(id) {
    
    x <- selected_res[[id]]
    
    if (length(x$media) == 0) {
      return(NA_character_)
    }
    
    x$media[[1]][[id]][[1]]$identifier
  }
)

image_urls_df <- tibble(
  key = names(image_urls),
  image_url = unname(image_urls)
)

sampled <- sampled %>%
  left_join(image_urls_df, by = "key")

sampled %>%
  summarise(
    total = n(),
    with_url = sum(!is.na(image_url)),
    without_url = sum(is.na(image_url))
  )

# download media ----
library(httr2)

download_image <- function(url, path) {
  
  tryCatch({
    
    request(url) %>%
      req_timeout(60) %>%
      req_perform() %>%
      resp_body_raw() %>%
      writeBin(path)
    
    TRUE
    
  }, error = function(e) {
    FALSE
  })
}

sampled <- sampled %>%
  mutate(
    image_id = sprintf("HERB_%04d", row_number()),
    image_path = file.path(
      "sample_images",
      paste0(image_id, ".jpg")
    )
  )

sampled$downloaded <- mapply(
  download_image,
  sampled$image_url,
  sampled$image_path
)

table(sampled$downloaded)

# retrieve and download floral pool ----
candidate_magnoliopsida <- candidate %>% 
  filter(class == "Magnoliopsida")

flower_matches <- candidate_magnoliopsida %>%
  filter(if_any(
    where(is.character), ~ str_detect(str_to_lower(.x), "flower"))) %>% 
  filter(!key %in% sampled$key)

selected_flower_res <- occ_search(
  gbifId = flower_matches$key,
  mediaType = "StillImage",
  limit = 500
)

image_urls_flower <- sapply(
  names(selected_flower_res),
  \(id) {
    
    x <- selected_flower_res[[id]]
    
    if (length(x$media) == 0) {
      return(NA_character_)
    }
    
    x$media[[1]][[id]][[1]]$identifier
  }
)

flower_matches <- flower_matches %>%
  left_join(tibble(
    key = names(image_urls_flower),
    image_url = unname(image_urls_flower)
  ), by = "key")

flower_matches <- flower_matches %>%
  mutate(
    image_id = sprintf("FLOW_%04d", row_number()),
    image_path = file.path(
      "sample_images/maybe_flowers",
      paste0(image_id, ".jpg")
    )
  )

flower_matches$downloaded <- mapply(
  download_image,
  flower_matches$image_url,
  flower_matches$image_path
)

table(flower_matches$downloaded)
