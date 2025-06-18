
🛍️ Shop Smart Analysis and Customer Segmentation

📊 Overview

This project involves an end-to-end analysis of customer interactions and purchasing behavior on a shopping platform. The goal is to derive actionable business insights, identify high-value customers, and optimize sales and marketing strategies.


---

📚 Libraries Used

pandas - Data manipulation and analysis

numpy - Numerical computing

matplotlib - Data visualization

seaborn - Statistical data visualization

sklearn - Machine learning (clustering, model building)

KMeans - Clustering for customer segmentation



---

✅ Business Objectives

Clearly define a business task

Import data from a real dataset

Document all data cleaning steps

Analyze customer interactions, orders, and purchasing patterns

Create compelling visualizations to support insights

Summarize key findings

Make business recommendations

Publish the case study



---

📥 Dataset Handling

Loaded and explored the dataset to understand the structure and key features

Cleaned missing values, corrected inconsistent formats, and removed duplicates

Performed feature engineering where necessary (e.g., interaction duration, customer value labels)



---

📈 Key Findings

🔁 Uniform Distribution of Engagement

Customer engagement was generally stable until a significant spike on the 29th.

🔍 Action: Investigate the cause of the spike and optimize or replicate the actions that led to it for future campaigns.



---

🏆 Top Performing Products

Canon EOS R5 is the top-performing product:

~39,715 interactions

~$9.92M revenue

~2,231 units sold


In contrast:

Nintendo Switch had ~36,979 interactions and ~$567K revenue (1,891 units sold).

Coca-Cola 12-Pack had ~38,440 interactions but only ~$12.7K revenue (2,124 units sold).


💡 Recommendation: Leverage the popularity of the Canon EOS R5 with cross-promotions, bundled offers, or loyalty programs.



---

⏱️ Average Interaction Duration

Customers spend an average of 4 days interacting with the platform.

💡 Recommendation: Use this insight to optimize engagement and personalize user journeys within that window.



---

📦 Order Success Rates

✅ Successful Orders: ~33.3%

❌ Failed/Canceled Orders: ~66.7%

🚨 Urgent Recommendation: Address the high failure rate by:

Improving payment infrastructure

Offering better transaction feedback

Reducing steps to checkout

Providing 24/7 support for failed payments




---

💎 Customer Segmentation (KMeans)

Segment Labels:

Label [1]: High-value customers

Label [2]: Mid-value customers

Label [0]: Low-value customers


Top Customer: ID bb7fd0af spent ~$41,417.4

📣 Recommendation:

Target high-value customers with early access to products

Launch loyalty rewards for repeat buyers

Offer upsells to mid-tier customers

Retarget low-tier customers with special deals




---

💰 Revenue and Losses

Revenue from Successful Orders: ~$22.76M

Loss from Failed/Canceled Orders: ~$44.68M

🚨 Critical Insight: Losses are nearly double the earned revenue due to transaction failures.

🛠️ Recommendation: Urgently review payment and transaction systems to minimize failed checkouts.



---

🌍 Revenue by Location

Top Revenue Country: Singapore - ~$252,274.67

Lowest Revenue Country: Slovenia - ~$180.00

📣 Recommendation: Focus high-return campaigns in top-revenue regions, but explore potential reasons for low performance in regions like Slovenia and tailor market-specific campaigns.



---

🌐 Interactions by Location

Highest Interactions: South Korea - 6,997 interactions

Lowest Interactions: Christmas Island - 3,471 interactions

📣 Recommendation: Target high-interaction regions like Korea with new product launches and regional promotions to drive conversions.



---

📌 Recommendations Summary

🔍 Investigate causes of the activity spike around the 29th and replicate successful strategies.

🛍️ Focus marketing on best-selling products (e.g., Canon EOS R5) and top regions (e.g., Singapore, Korea).

💳 Improve transaction infrastructure to reduce ~66.7% order failure rate.

👥 Segment customers for targeted marketing — offer premium services and exclusive deals to top-tier users.

📈 Launch cross-selling and upselling campaigns based on behavior and segmentation.

🌐 Use geographical insights to localize marketing strategies and promotions.

📦 Improve product bundling and retargeting strategies for low-converting products like Coca-Cola packs.

