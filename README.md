# Polyratings TF IDF

This program will print out the top ten relevant ratings on the polyratings website. It does this using TF IDF with cosine similarity discussed in the slides and report.

## Development

The code uses the unified spark configuration that allows a json file to be loaded as a dataframe. Before the user is prompted for a query `ratingsTfIdf.count()` is called to force the main dataframe to decrease evaluation time once a user is prompted for a query. This allowed us to measure the average query response time to be under 10 seconds.
