'''
Rating Product & Sorting Reviews in Amazon
Business Problem: To calculate product ratings more accurately work and product reviews more accurately to sort.

Dataset Story: The user of the product with the most comments in the electronics category It has ratings and reviews.

Variables:

reviewerID : User ID
asin : Product ID
reviewerName : Username
helpful : Degree of Useful Evaluation (Sample : 2/3)
reviewText : Review (User-written review text)
overall : Product Rating
summary : Evaluation Summary
unixReviewTime : Evaluation Time
reviewTime : Review Time
day_diff : Number of Days Since Evaluation
helpful_yes : The Number of Times the Review was Found Helpful
total_vote – Number of Votes Given to the Review
'''

# Let's load the dataset.
import pandas as pd
import math
import scipy.stats as st
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
df = pd.read_csv("datasets/amazon_review.csv")
df.head()

# Let's summarize the dataset.
df.info()

# Task 1: Calculate the Average Rating according to the current comments and compare it with the existing average rating.
df["overall"].mean()

# We found the overall score average, but when calculating it, it could be the votes from 4 years ago.
# Votes cast 4 years ago may be out of date. Instead, we need to find the recent weighted average.
# The reason we use the time-weighted average is because we want to give a coefficient to the most recently cast votes.
# Because the last votes represent the product up to date. For example, there may have been a problem with deliveries recently.
# That's why people will give it a lower rating. This scoring method will represent the product much better for customers.

df.loc[df["day_diff"] <= 30, "overall"].mean() * 28 / 100 + \
df.loc[(df["day_diff"] > 30) & (df["day_diff"] <= 90), "overall"].mean() * 26 / 100 + \
df.loc[(df["day_diff"] > 90) & (df["day_diff"] <= 180), "overall"].mean() * 24 / 100 + \
df.loc[(df["day_diff"] > 180), "overall"].mean() * 22 / 100

# day_diff → Represented the number of days since the evaluation.
# That's why we gave more coefficients to those with a small day_diff so that it has a greater effect on the result.
# The largest day_diff ones show a very distant date, so we lowered their coefficients.
# These coefficients may vary from person to person. As can be seen, there was an increase of approximately 0.11 in time-weighted scoring.


# Task 2: Determine the 10 reviews that will be displayed on the product detail page for the product.

# We will use the Wilson Lower Bound method. In short, we can say that this method works according to the logic of liked/disliked, useful/not useful.
# helpful_yes → indicated that the review was found useful. To use the Wilson Lower Bound method, let's create a variable called helpful_no.

# Total Votes — Useful Votes = Useless Votes
df["helpful_no"] = df["total_vote"]-df["helpful_yes"]

# Now let's create the dataframe from the helpful_yes and heplful_no variables.
comments = pd.DataFrame({"up": df["helpful_yes"], "down": df["helpful_no"]})

# Now let's create our wilson_lower_bound function and apply it to our comments variable.
def wilson_lower_bound(up, down, confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

comments["wilson_lower_bound"] = comments.apply(lambda x: wilson_lower_bound(x["up"],x["down"]),axis=1)

# Finally, let's see our top 10 reviews.
comments.sort_values("wilson_lower_bound", ascending=False).head(10)


# As a result: We have reached the top 10 reviews that best represent the product.
# We have created the ranking not according to the highest scores, but according to the reviews that best represent the product.
# A much more objective ranking.