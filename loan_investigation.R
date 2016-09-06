library(tm)
library(SnowballC)
library(dplyr)

setwd("C:/DataSciSideProjects/")

loans_a <- read.csv("LoanStats3a_securev1.csv", stringsAsFactor=FALSE, skip=1)
loans_b <- read.csv("LoanStats3b_securev1.csv", stringsAsFactor=FALSE, skip=1)
loans_c <- read.csv("LoanStats3c_securev1.csv", stringsAsFactor=FALSE, skip=1)
loans_d <- read.csv("LoanStats3d_securev1.csv", stringsAsFactor=FALSE, skip=1)

loans <- bind_rows(list(loans_a, loans_b, loans_c, loans_d))

loans_narrow[duplicated(loans_narrow$member_id), ]$id

rm(loans_a)
rm(loans_b)
rm(loans_c)
rm(loans_d)

#loans_a_narrow <- select(loans_a, id:pymnt_plan, desc:title)
#loans_a_narrow[loans_a_narrow$loan_status == 'Does not meet the credit policy. Status:Charged Off', ] <- 'Charged Off'
#loans_a_narrow[loans_a_narrow$loan_status == 'Does not meet the credit policy. Status:Fully Paid', ] <- 'Fully Paid'
#table(loans_a_narrow$loan_status)
#loans_a_narrow <- loans_a_narrow[loans_a_narrow$loan_status != '', ]
#prop.table(table(loans_a_narrow$loan_status))
#loans_a_narrow_desc <- filter(loans_a_narrow, desc != '')

loans_narrow <- select(loans, id:pymnt_plan, desc:title)
loans_narrow[loans_narrow$loan_status == 'Does not meet the credit policy. Status:Charged Off', ] <- 'Charged Off'
loans_narrow[loans_narrow$loan_status == 'Does not meet the credit policy. Status:Fully Paid', ] <- 'Fully Paid'
table(loans_narrow$loan_status)
loans_narrow <- loans_narrow[loans_narrow$loan_status != '', ]
prop.table(table(loans_narrow$loan_status))
loans_narrow_desc <- filter(loans_narrow, desc != '')

corpus <- Corpus(VectorSource(loans_narrow_desc$desc))
corpus <- tm_map(corpus, tolower)
corpus <- tm_map(corpus, PlainTextDocument)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stemDocument)

frequencies <- DocumentTermMatrix(corpus)
findFreqTerms(frequencies, lowfreq=20)
sparse <- removeSparseTerms(frequencies, .80)
desc_sparse <- as.data.frame((as.matrix(sparse)))

desc_sparse$loan_status <- loans_narrow_desc$loan_status

prop.table(table(filter(desc_sparse, year > 0)$loan_status))

rm(loans)
rm(corpus)
rm(frequencies)
rm(sparse)

library(caTools)
set.seed(144)
split <- sample.split(desc_sparse$loan_status, SplitRatio = 0.7)
train <- subset(desc_sparse, split)
test <- subset(desc_sparse, !split)

library(rpart)
library(rpart.plot)

desc_CART <- rpart(loan_status ~ ., data = desc_sparse, method = "class")
prp(desc_CART)

library(caret)
library(e1071)

num_folds <- trainControl(method = "cv", number = 10)
cp_grid <- expand.grid(.cp = seq(0.01, 0.5, 0.01))
train(loan_status ~ ., data = train, method = "rpart", trControl = num_folds, tuneGrid = cp_grid)
