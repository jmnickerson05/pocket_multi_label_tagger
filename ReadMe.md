Project: Write a Data Science Blog Post

* Business Understanding
    * Brief description:
        * Bookmarked and topically tagged Pocket App API data
    * Question 1:
        - What are my core technical interests?
    * Question 2
        - How have my technical interests changed over the years?
    * Question 3 
        - What keywords are the most important for my leading topical tags and do they reveal additional insights?
* Data Understanding
    * Access and Explore:
        - The data consists of saved articles along with a series of metadata attributes such as topical tags, word count, image and video links, example text data, etc..
* Prepare Data
    * Clean:
        - Download full HTML body text
        - Remove non-ascii characters
        - Remove stop words
* Data Modeling (Optional)
    * Fit model:
        - Multi-Label Classification (NOTE: In repo, but not discussed in this assignment)
    * validate the model
        - Again, in the repo but not discussed in the Blog post
* Evaluation
    * Question 1
        * Analyze:
            - Simple count of tagged articles (combination of manually tagged and tagged by NLP model)
        * Visualize:
            - models/tag_counts.png
        * Brief explanation for visualisation:
            - Discussion of misleading nature of simplistic visualizations
    * Question 2
        * Analyze
            - Changes in tags over time
        * Visualize
            - analysis/growth_over_time.png
            - tag_race.gif
        * Brief explanation for visualisation
            - Tells the story of my journey as a data practitioner
    * Question 3
        * Analyze
            - TFIDF keywords of text of articles associated with top tags
        * Visualize
            - analysis/wordcloud.png
        * Brief explanation for visualisation
            - Demonstrates deeper insights. I.e. keyword "SQL" is higher than "Python" even though the reverse is
            true of overall tag counts. Conclusion -- SQL is more ubiquitious, but is often mentioned secondarily
            and not as a core topic. 