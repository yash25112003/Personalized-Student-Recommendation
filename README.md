# Personalized-Student-Recommendation

This project is a comprehensive quiz performance analysis tool that processes quiz data, extracts insights, and provides recommendations for improvement. It leverages Python libraries such as requests, numpy, matplotlib, rich, and sklearn to analyze quiz results, compare historical data, and visualize trends.

****

## Setup Instructions

**Clone this repository:**

```bash
   git clone https://github.com/yash25112003/Personalized-Student-Recommendation.git
```
**Step 1: Create a Virtual Environment**

```bash
  # For Windows
  python -m venv venv
  venv\Scripts\activate

  # For macOS/Linux
  python3 -m venv venv
  source venv/bin/activate
```
**Step 2: Install Required Libraries**

```bash
  pip install -r requirements.txt
```

**Step 3: Run the Script**
```bash
  cd Personalized-Student-Recommender/
  python3 main.py
```

## Features

This program will fetch the latest quiz data and historical records, analyze them, and display an enhanced performance report.
Visual trend charts will be generated to help track improvement over time.

## Key Features
- **Fetch Quiz Data:** Retrieves quiz performance and historical data from APIs.

- **Analyze Performance:** Computes accuracy, improvement trends, and mistake correction rates.

- **Compare Historical Data:** Identifies trends in accuracy, score, and percentile ranking.

- **Generate Insights & Recommendations:** Suggests improvement strategies based on performance metrics.

- **Visualize Trends:** Uses Matplotlib to create performance trend graphs.

- **Display Reports:** Uses rich to present structured and color-coded results in the terminal.



## API Configuration

Update the API keys in the script to fetch quiz performance and historical data:


| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `quiz_performance_api_key` | `string` | https://api.jsonserve.com/XgAgFJ |
| `historical_performance_api_key`      | `string` | https://api.jsonserve.com/XgAgFJ |



## Screenshots

![App Screenshot](https://github.com/user-attachments/assets/e532bb63-f78c-4835-8c1f-1c9982ff8b67)

![App Screenshot](https://github.com/user-attachments/assets/48ede3aa-2bd7-4178-8060-b24b5a4f8fe5)


![App Screenshot](https://github.com/user-attachments/assets/6a93d625-0327-4959-84b2-0e3e385505ca)

![App Screenshot](https://github.com/user-attachments/assets/d4feab6e-b970-4124-8f0a-2ca0a98ac3e0)


