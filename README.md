# COS 783 Final Project
Topic 2:<br>
 Metadata Analysis: AI techniques can assist in analyzing large volumes of metadata by automating pattern recognition, anomaly detection, and correlation analysis. AI algorithms can identify suspicious patterns or outliers that may indicate important evidence.

## Edwin Sen-Hong Chang 20424575
Implementation of various anomaly detection models:
- Credit Card Fraud - The dataset that was worked on was on credit card transactions made by European cardholders in September 2007. Due to confidentiality issues, features are anonymized. This gives us a dataset of unlabeled data which will fall under unsupervised learning. We made use of Isolation Forest module that exists within the Scikit-Learn library in Python to do the anomaly detection.
- Phishing URLs - The dataset is of URLs that may or may not be classified as a Phishing site. having_IP_Address, URL_Length,Shortining_Service, having_At_Symbol, double_slash_redirecting, Prefix_Suffix, having_Sub_Domain, SSLfinal_State, Domain_registeration_length, Favicon, port, HTTPS_token,R equest_URL, URL_of_Anchor, Links_in_tags, SFH, Submitting_to_email, Abnormal_URL,Redirect,on_mouseover, RightClick, popUpWidnow, Iframe, age_of_domain, DNSRecord, web_traffic, Page_Rank, Google_Index, Links_pointing_to_page and Statistical_report are the features of the dataset. The anomaly detection was implemented with supervised learning, using Random Forest Classifiers provided by Scikit-Learn library in Python.

## Morgan Bentley 18103007