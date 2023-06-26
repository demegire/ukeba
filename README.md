# METU Industrial Engineering System Design Project `Mobile Game Marketing Budget Allocation Decision Support System`

by Büşra Ahıskalıoğlu, Ege Demir, Kaan Serdar, Alparslan Sertel and Utku Uğur Yağcı
Project Sponsor: Ufuk Serkan Yıldırım representing The Game Circle
Academic Advisors: Sinan Gürel, Cem İyigün

This is the code for the senior capstone project `Mobile Game Marketing Budget Allocation Decision Support System`. Below are the executive
summary and usage instructions for this repo. Access the full article [here](https://demegire.github.io/IE498_Report.pdf).

# Executive Summary
## The Problem
Game publisher companies are responsible for a range of decision-making processes, including the
allocation of budget to various social media platforms, and determination of the optimal budgeting
strategy to maximize profit from a given game across different platforms. However, currently used
methods for determining these strategies rely on past campaign experience of decision makers and are
far from a rule-based and reliable budget allocation decision-making mechanism in most companies.

## Proposed Solution
The problem is tackled wıth a framework composed of three parts: daily budgeting for different
audiences in a platform, the budgeting between the platforms, and revenue estimation. This is a
modular approach and is useful in a number of scenarios where the budget is allocated for:
* a single platform and a single audience in that platform,
* a single platform and multiple audiences in that platform and
* multiple platforms, multiple audiences.

All of the models directly use the data coming from previous campaigns in advertising platforms. The
multiplatform budgeting model solves a knapsack problem that maximizes profit by distributing the
total campaign budget between the platforms. For each platform, the daily budgeting model leverages
optimal control theory to find the optimal campaign (platform budget-app installs-campaign length)
triplets. For each triplet, a revenue estimation is made using multilinear regression. Then the user
selects a campaign depending on the projected ROI and the triplet parameters. Finally the user is
presented with a daily budget for each platform, which is updated daily as new data is acquired from
the advertising platforms. This solution is packaged in a ready to use, user friendly software package
which can be accessed in this repo.

## Results
The solution is put to test in a single platform and a single audience environment. Results showed that
the actual total number of installs is 47% higher than the projected number of installs. Although a
favorable outcome, one potential factor contributing to this inaccuracy is the continuous updates and
improvements made by The Game Circle in its ad designs where the data fed into the model reflects
an outdated situation..
In conclusion, our project contributes to the field of multiplatform online advertising by providing a
comprehensive decision support system that addresses various aspects of campaign budget allocation.
Further experiments under different settings could be made to further validate the effectiveness of our
approach.

# Instructions
