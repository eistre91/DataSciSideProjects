library(rjson)
library(plyr)
library(dplyr)
library(tm)
library(ggplot2)

json_file_business <- "yelp_academic_dataset_business.json"
json_file_review <- "yelp_academic_dataset_review.json"

con <- file(description=json_file_review, open="r")
con_business <- file(description=json_file_business, open="r")

#2225213 reviews

# Take 200,000 of the reviews.
df <- data.frame()
for(i in 1:2000) {
  n <- 100
  lines <- readLines(con, n)
  converted <- lapply(lines, fromJSON)
  df_tmp <- ldply(converted, data.frame, stringsAsFactors=FALSE)
  df <- plyr::rbind.fill(df, df_tmp)
  print(i)
}

# Load the first 10,000 businesses.
n <- 10000
lines <- readLines(con_business, n)
converted <- lapply(lines, fromJSON)

close(con)
close(con_business)

# Create a corpus for the reviews and pre-process.
corpus <- Corpus(VectorSource(df$text))
corpus <- tm_map(corpus, tolower)
corpus <- tm_map(corpus, PlainTextDocument)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stemDocument)

frequencies <- DocumentTermMatrix(corpus)

keywords_hist <- list()

keywords_hist['romant'] = sum(inspect(frequencies[, grepl("romant", frequencies$dimnames$Terms)]))
keywords_hist['relax'] = sum(inspect(frequencies[, grepl("relax", frequencies$dimnames$Terms)]))
keywords_hist['excit'] = sum(inspect(frequencies[, grepl("excit", frequencies$dimnames$Terms)]))
keywords_hist['cozy'] = sum(inspect(frequencies[, grepl("cozy", frequencies$dimnames$Terms)]))
keywords_hist['prett'] = sum(inspect(frequencies[, grepl("prett", frequencies$dimnames$Terms)]))
keywords_hist['intima'] = sum(inspect(frequencies[, grepl("intima", frequencies$dimnames$Terms)]))
keywords_hist['advent'] = sum(inspect(frequencies[, grepl("advent", frequencies$dimnames$Terms)]))
keywords_hist['vegetarian'] = sum(inspect(frequencies[, grepl("vegetarian", frequencies$dimnames$Terms)]))
keywords_hist['date'] = sum(inspect(frequencies[, grepl("^date", frequencies$dimnames$Terms)]))

test_df <- data.frame(keywords_hist)
test_df <- t(test_df)
test_df <- data.frame(test_df)
colnames(test_df) <- c("count")

ggplot(test_df) +
  geom_bar(aes(x=rownames(test_df), y=log(count), fill="red"), stat="identity") +
  guides(fill=FALSE) + 
  xlab("Keywords") +
  ylab("Log Count of Keyword Occurences")

list_of_businesses <- sapply(converted, "[[", "business_id")

date_places <- filter(df, grepl("date", text))
date_places <- filter(date_places, business_id %in% list_of_businesses) 
list_of_date_places <- unique(date_places$business_id)

date_place_info <- converted[which(sapply(converted, "[[", "business_id") %in% list_of_date_places)]
date_place_categories <- sapply(date_place_info, "[[", "categories")
num_date <- length(unique(unlist(date_place_categories)))

romant_places <- filter(df, grepl("romant", text))
romant_places <- filter(romant_places, business_id %in% list_of_businesses) 
list_of_romant_places <- unique(romant_places$business_id)

romant_place_info <- converted[which(sapply(converted, "[[", "business_id") %in% list_of_romant_places)]
romant_place_categories <- sapply(romant_place_info, "[[", "categories")
num_romant <- length(unique(unlist(romant_place_categories)))

categories_df <- data.frame(x = c("date", "romant"), y = c(num_date, num_romant))

ggplot(categories_df) +
  geom_bar(aes(x=x, y=y, fill="red"), stat="identity") +
  guides(fill=FALSE) + 
  xlab("Keywords") +
  ylab("Unique Categories Under Keyword")

#ambianc
#atmospher
#authent
#band
#beauti
#blue
#decor
#downtown
#fanci
#french
#recent
#reserv
#southern
#spice
#sport
#surpris
#vegetarian
#view
#weird

#pretty
#date
#romantic
#adventure
#exciting
#fancy
#relaxing
#cozy
#intimate
#unique
#quiet
#loud

#need two plots
  #basic keyword analysis on reviews, what's out there
  #variance in keywords used by business category

#tailoring the night to them

#give parameters and let yelp suggest a complete day
  #parameters to consider
    #price
    #type of evening
    #distance
#benefits
  #companies can opt in to offer special experience discounts
    #they are exposed to people who may not have seen them before but who like what they have to offer
  #get yelp more active in people's lives
#collaborative filtering
  #users review more than one thing
    #check
#keyword analysis to find out what category something is
  #i.e. romantic
  # adventurous
  # exciting
#price for the experience
#need a date idea
