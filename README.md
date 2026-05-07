# AI Detector Hypothesis Testing
Submitted as the final project for my senior year Statistics* course, this project performs two hypothesis tests to determine whether Sapling AI (a free AI content detection API) can sucessfully detect whether AI was used to generate a selection of 100 words.
Our data collection process involves using AI to lightly edit, heavily edit, and completely generate a new essay based on 5 newspaper articles written by high school students.
Each essay is assigned an expected percentage of AI-generated content based on the process used to create it: 0%, 25%, 75%, and 100%. Essays are also split into 6 chunks of 100 words each before being processed through Sapling AI. 
To test whether the detector performed well on the binary classification task (>50% AI used), we ran a one-proportion z-test. Additionally, we tested the accuracy of Sapling AI's 
AI score by using a t-test to compare the difference in means between the Sapling score and the expected percentage bucket. Ours findings showed that while Sapling AI tended to overshoot 
score predicitons, it could sucessfully determine with statistical significance whether an essay was completely generated or heavily edited with AI. 
