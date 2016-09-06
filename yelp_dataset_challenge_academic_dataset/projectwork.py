import pandas as pd
import json
from collections import defaultdict
import pickle

#step 1: read in all of the data for one city from the dataset

def find_city_with_most():
#subsetting the data, which city has the most listed businesses
	businesses = []
	city_counts = defaultdict(int)
	with open("yelp_academic_dataset_business.json") as f:
		for line in f:
			parsed = json.loads(line)
			city_counts[parsed['city']] += 1
			businesses.append(parsed)
			
	sorted(city_counts.items(), key = lambda k_v: k_v[1])[-1]

def pickle_vegas_data():
	las_vegas_businesses = []
	with open("yelp_academic_dataset_business.json") as f:
		for line in f:
			parsed = json.loads(line)
			if parsed['city'] == 'Las Vegas':
				las_vegas_businesses.append(parsed)
				
	print(len(las_vegas_businesses))

	business_ids = set()
	for entry in las_vegas_businesses:
		business_ids.add(entry['business_id'])
		
	print(len(business_ids))

	las_vegas_reviews = []
	with open("yelp_academic_dataset_review.json") as f:
		for line in f:
			parsed = json.loads(line)
			if parsed['business_id'] in business_ids:
				las_vegas_reviews.append(parsed)
				
	print(len(las_vegas_reviews))

	with open('las_vegas_reviews.pickle', 'wb') as f:
		pickle.dump(las_vegas_reviews, f, pickle.HIGHEST_PROTOCOL)
		
	with open('las_vegas_businesses.pickle', 'wb') as f:
		pickle.dump(las_vegas_businesses, f, pickle.HIGHEST_PROTOCOL)
		
def pickle_user_data(review_list):
	
	user_ids = set()
	for entry in review_list:
		user_ids.add(entry['user_id'])
		
	print(len(user_ids))
	
	las_vegas_users = []
	with open("yelp_academic_dataset_user.json") as f:
		for line in f:
			parsed = json.loads(line)
			if parsed['user_id'] in user_ids:
				las_vegas_users.append(parsed)
				
	print(len(las_vegas_users))
	
	with open('las_vegas_users.pickle', 'wb') as f:
		pickle.dump(las_vegas_users, f, pickle.HIGHEST_PROTOCOL)
						
def build_set_of_categories(business_list):
	categories = set()
	for entry in business_list:
		for category in entry['categories']:
			categories.add(category)
	return categories	

def build_set_of_attributes(business_list):
	attributes = set()
	for entry in business_list:
		for attribute in entry['attributes'].keys():
			attributes.add(attribute)
	return attributes
	
def build_set_of_ambiences(business_list):
	ambiences = set()
	for entry in business_list:
		for attribute in entry['attributes'].keys():
			if attribute == 'Ambience':	
				for ambience in entry['attributes']['Ambience'].keys():
					ambiences.add(ambience)
	return ambiences
	
def list_of_businesses_with_ambience_attribute(business_list):
	businesses = []
	for entry in business_list:
		if 'Ambience' in entry['attributes'].keys():
			businesses.append(entry)
	return businesses
	
#def dict_of_businesses_to_review_stars(business_df, review_df):
#	business_review_dict = {}
#	for entry in business_list:
#		entry['business_id'] = 
	
with open('las_vegas_reviews.pickle', 'rb') as f:
    las_vegas_reviews = pickle.load(f)	
print("loaded reviews")
	
with open('las_vegas_businesses.pickle', 'rb') as f:
    las_vegas_businesses = pickle.load(f)
print("loaded businesses")
	
with open('las_vegas_users.pickle', 'rb') as f:
	las_vegas_users = pickle.load(f)
print("loaded users")
	
reviews_df = pd.io.json.json_normalize(las_vegas_reviews)
print("loaded reviews df")

users_df = pd.io.json.json_normalize(las_vegas_users)
print("loaded users df")

businesses_df = pd.io.json.json_normalize(las_vegas_businesses)
print("loaded businesses df")
	
categories = build_set_of_categories()
attributes = build_set_of_attributes()
ambiences = build_set_of_ambiences()

business_dict = pd.io.json.json_normalize(las_vegas_businesses)
restaurant_bools = test['categories'].apply(lambda x: 'Restaurants' in x)

not_ambient_restaurants = df['categories'].apply(lambda x: 'Restaurants' not in x)
df['categories'][not_ambient_restaurants]

list_of_cats = df['categories'].values
set([cat for list in list_of_cats for cat in lists])

#how can I parse out the data to prepare for collaborative filtering

#how evenly distributed are the user reviews
#i.e. anything like 10% of the users write 90% of the reviews
users_df['review_count'].sum()
len(users_df['review_count']) # 256148
users_df['review_count'].sort_values(ascending=False)[0:25613].sum()/users_df['review_count'].sum()
#first 25613 (10%) make up 66% of the reviews
#first 50% of users make up 96% of the reviews

#how many users have less than 2 reviews
#need at least two to test the collaborative filtering engine
sum(users_df.review_count < 2) #31183

users_with_reviews = users_df[users_df.review_count >= 2]

#how many reviews do businesses tend to have
businesses_df['review_count'].quantile(.9)
#90% of businesses have 110 reviews or less

#frequency of occurence per category?
category_count = defaultdict(int)
for category_list in businesses_df['categories']:
	for category in category_list:
		category_count[category] += 1

sorted(category_count.items(), key=lambda x: x[1])
category_count_df = pd.Series(category_count)

#business categories that tend to occur together?
#only 266 businesses of the 17423 las vegas have 0 or 1 categories
#754 categories, thus 283881 pairings
#how many are actually non-zero pairings?
#only 5425 non-zero pairings, about 1.9%

#combo_count_dict = defaultdict(int)
#x = 0
#businesses_df_combos = businesses_df[businesses_df['categories'].apply(lambda x: len(x) > 1)]
#sample_size = math.floor(len(businesses_df_combos)/10)
#businesses_df_combos = businesses_df_combos.sample(sample_size)
#for combo in itertools.combinations(categories, 2):
#	x += 1
#	print(x)
#	temp = businesses_df[businesses_df['categories'].apply(lambda x: (combo[0] in x) and (combo[1] in x))]
#	combo_count_dict[combo] = len(temp)

combo_count_dict = defaultdict(int)
businesses_df_combos = businesses_df[businesses_df['categories'].apply(lambda x: len(x) > 1)]
x = 0
for cat_list in businesses_df_combos['categories']:
	for combo in itertools.combinations(cat_list,2):
		combo_count_dict[tuple(sorted(combo))] += 1
	x += 1
	print(x)
	
combo_count_series = pd.Series(combo_count_dict)	 
combo_count_series.loc[(slice('ATV Rentals/Tours',"Women's Clothing"), 'Yoga')]
	
#2524 businesses of the las vegas listed are closed


	
#compute user distance
#we can recommend places to users by segmenting users
#collaborative filtering
#given a user A, find users that fit well with A by similarity of ratings
	#take a user A
	#for each review of A, take business_id
	#filter reviews_df for reviews of business_id
	#take the users with reviews of similar scores to A
	#find places that occur in non-trivial rates among those users
	#compute average star ratings for those places
	#serve up the sorted recommendations for A
	
def predict_new_ratings_for_user(user_id, reviews_df, categories_to_recommend = None, star_margin = .5):
	user_reviews = reviews_df[reviews_df['user_id'] == user_id]
	reviewed_by_user = user_reviews[['business_id', 'stars']]
	
	additional_reviews = reviews_df[(reviews_df['business_id'].isin(reviewed_by_user['business_id']))]
	#remove the reviews that came from the user in question
	additional_reviews = additional_reviews[additional_reviews['user_id'] != user_id]	
	additional_reviews = pd.merge(additional_reviews, reviewed_by_user, how='inner', on='business_id', suffixes=('_other', '_user'))
	additional_reviews = additional_reviews[((additional_reviews['stars_other'] - star_margin) < additional_reviews['stars_user']) & \
											((additional_reviews['stars_other'] + star_margin) > additional_reviews['stars_user'])]
	
	if len(additional_reviews) < 10:
		print("Not enough information to recommend for user.")
		return {}
	
	#find places regularly reviewed by the other users remaining in the list
	other_reviews = reviews_df[reviews_df['user_id'].isin(additional_reviews['user_id'])]
	#remove businesses that were already reviewed by the original user
	other_reviews = other_reviews[~other_reviews['business_id'].isin(additional_reviews['business_id'])]
	#getting counts of the reviews of businesses among the other users
	review_counts = other_reviews['business_id'].value_counts()
	review_counts = review_counts[review_counts >= 10]
	other_reviews = other_reviews[other_reviews['business_id'].isin(review_counts.index)]

	if len(other_reviews) == 0:
		print("Not enough information to recommend for user.")
		return {}
	
	ratings_guess = other_reviews.groupby('business_id')['stars'].mean()
	ratings_guess = ratings_guess.sort_values(ascending = False)
	#predictions = businesses_df[businesses_df['business_id'].isin(ratings_guess.index)].join(ratings_guess, on='business_id', rsuffix='_pred')	
	
	return ratings_guess

#still need to test whether this is accurate at all	
	#baseline predictions will be the overall average for a business
	#test whether the average computed above or the baseline is closer to the user's actual rating
	
def test_prediction(reviews_df, n=100):
	#sample reviews_df for users with at least 20 reviews
	reviews_counts = reviews_df['user_id'].value_counts()
	reviews_counts = reviews_counts[reviews_counts >= 50]
	if n > len(reviews_counts):
		n = len(reviews_counts)
	
	sampled_users = list(reviews_counts.sample(n).index)
	
		
	
	#remove 8 reviews from those
	sampled_reviews = reviews_df[reviews_df['user_id'].isin(sampled_users)]
	groups_by_id = sampled_reviews.groupby('user_id')
	random.seed(42)
	sampled_indices = list(map(lambda x: random.sample(list(x[1]), math.floor(len(x[1]) * .6)), groups_by_id.indices.items()))
	sampled_indices = [index for sublist in sampled_indices for index in sublist]
	sampled_reviews_train = sampled_reviews.iloc[sampled_indices]
	sampled_reviews_test = sampled_reviews[~sampled_reviews.index.isin(sampled_reviews_train.index)]
	
	print(len(sampled_reviews_train) + len(sampled_reviews_test) == len(sampled_reviews))
	
	remove_test_data_df = reviews_df[~reviews_df.index.isin(sampled_reviews_test.index)]
	
	print(len(reviews_df) - len(remove_test_data_df) == len(sampled_reviews_test))
	
	#calculate predictions
	predictions_by_user = {}
	for i in range(n):
		predictions_by_user[sampled_users[i]] = predict_new_ratings_for_user(sampled_users[i], remove_test_data_df)
	
	#match predictions with removed reviews
	sampled_reviews_test_matched_list = []
	for i in range(n):
		current_user = sampled_users[i]
		businesses_with_predicted_ratings = predictions_by_user[current_user].to_frame().reset_index(level=0)
		#businesses_with_predicted_ratings = predictions_by_user[current_user].index
		current_user_entries = sampled_reviews_test[sampled_reviews_test['user_id'] == current_user]
		#current_user_matched_businesses = current_user_entries[current_user_entries['business_id'].isin(businesses_with_predicted_ratings)]
		current_user_matched_businesses = current_user_entries.merge(businesses_with_predicted_ratings, how='inner', on='business_id', suffixes=['_actual', '_pred'])
		sampled_reviews_test_matched_list.append(current_user_matched_businesses)
	sampled_reviews_test_matched = pd.concat(sampled_reviews_test_matched_list)
	
	#append baseline rating
	businesses_averages = businesses_df[['business_id', 'stars']]
	businesses_averages.columns = ['business_id', 'stars_average']
	sampled_reviews_test_matched = sampled_reviews_test_matched.merge(businesses_averages, how='inner', on='business_id')
	
	#calculate error in engine prediction from actual rating
	mean_error_pred = ((sampled_reviews_test_matched['stars_actual'] - sampled_reviews_test_matched['stars_pred']) ** 2).mean()
	#calculate error in baseline prediction (the average business rating) from actual rating
	mean_error_baseline = ((sampled_reviews_test_matched['stars_actual'] - sampled_reviews_test_matched['stars_average']) ** 2).mean()
	
	#t-test to see if error engine prediction is significantly less than baseline
	#cohen's d for effect size calculation
	
	print("Prediction error: " + str(mean_error_pred))
	print("Baseline error: " + str(mean_error_baseline))
	
	return
	

	
	
#not yet restricting the restrictions I give by category
	#i.e. users that like the same doctors are currently assumed to also like the same restaurants
	
	
#what happens if there are no other reviews
	#only consider businesses with more than 1 review
		#presumably you can't leave multiple reviews for a place	
	
#other_reviews.groupby('business_id')['business_id'].count()
	
#compute business distance
	#categorical similarity (based on existent categories in dataset)
	#measuring categorical distance as the length of the set of differences of the two lists of categories
	#normalize by the length of the starting set of categories
def calculate_categorical_distances(business_id, businesses_df):
	#current_categories = list(businesses_df[businesses_df['business_id'] == business_id]['categories'])[0]
	current_categories = set(businesses_df[businesses_df['business_id'] == business_id]['categories'].iloc[0])
	
	categorical_distances = businesses_df['categories'].apply(lambda x: len(current_categories - set(x))/len(current_categories))
	categorical_distances = pd.concat([businesses_df['business_id'], categorical_distances], axis=1)
	categorical_distances = categorical_distances[categorical_distances['business_id'] != business_id]
	
	return categorical_distances
	
	#purpose/theme similarity (based on review text analysis)
	#characteristically similar, i.e. are the places loud? pretty to look at?

import nltk
	
def find_characteristically_similar_businesses(business_id, businesses_df, reviews_df):
	return

def find_businesses_by_characteristic(business_id, businesses_df, reviews_df):
	return
	
	#we can recommend users to places by segmenting places
#content filtering 
#use all attributes, categories coincidence

#categorical similarity
#given a particular business, 
#filter the database of businesses to those which have the same categories
#then proceed from existing a user's reviews to suggest categorically similar places


#what can I learn about the general content of the review text

#average review length

#think about how yelp makes money to help guide project
#think about user engagement

#heriarchical clustering
#k-means clustering