# Techn_task


## This is my solution:


Data Science part:

1. A report/notebook that explains the problem ... - submission-nb.ipynb

2. A clearly written .py .... - train_and_pred.py

3. Predictions in a file .... - test_labels.csv



MLOps part:

I made several changes to the default files: Makefile, Dockerfile, and pyproject.toml. All changes and the reasons for them you can find in SUBMISSION.md.

Deployment:

I wasn't sure how you wanted me to submit the results — I suspected you might want source code (for example, via a GitHub repository) because the Makefile includes commands like make clean. To be safe, I’ve also provided a Docker image on Docker Hub. I also deployed it to AWS App Runner in order to practice and so you can test it directly.

kertn/inference_api:latest - Docker image in Docker Hub

https://hn3jwxqztv.us-east-1.awsapprunner.com/docs#/  - deployed image using AWS App Runner

https://github.com/Kertn/MlOps_task - Source repos in GitHub



Best Regards,
Serhii Chepurko
