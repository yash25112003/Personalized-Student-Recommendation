import requests
import json
from datetime import datetime
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich import print as rprint
from rich.progress import track
from rich.columns import Columns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize the console for rich text formatting
console = Console()

# API keys for quiz performance and historical performance datasets
quiz_performance_api_key = "https://api.jsonserve.com/XgAgFJ"
historical_performance_api_key = "https://api.jsonserve.com/XgAgFJ"

def get_quiz_performance_data(api_key: str) -> Dict:
    """Fetch quiz performance data from the API with error handling."""
    try:
        response = requests.get(api_key)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list):
                return data[0] if data else {}
            else:
                return data
        else:
            console.print(f"Failed to retrieve quiz performance data. Status code: {response.status_code}")
            return {}
    except Exception as e:
        console.print(f"Error retrieving quiz performance data: {e}")
        return {}

def get_historical_performance_data(api_key: str) -> List[Dict]:
    """Fetch historical performance data from the API with error handling."""
    try:
        response = requests.get(api_key)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list):
                return data
            else:
                return [data]
        else:
            console.print(f"Failed to retrieve historical performance data. Status code: {response.status_code}")
            return []
    except Exception as e:
        console.print(f"Error retrieving historical performance data: {e}")
        return []

def analyze_quiz_performance(current_quiz: Dict, historical_quizzes: List[Dict]) -> Dict:
    """
    Analyze quiz performance and generate detailed output.
    This function processes both current and historical quiz data to provide insights.
    """
    # Process current quiz metrics
    current_accuracy = float(current_quiz['accuracy'].strip(' %'))
    current_topic = current_quiz['quiz']['topic'].strip()  # Clean topic name

    # Calculate topic-wise accuracy with proper improvement tracking
    topic_accuracy = {}

    # First, process historical quizzes to establish baseline
    for quiz in historical_quizzes:
        topic = quiz['quiz']['topic'].strip()  # Clean topic name
        accuracy = float(quiz['accuracy'].strip(' %'))

        if topic not in topic_accuracy:
            topic_accuracy[topic] = {
                'accuracies': [accuracy],
                'scores': [quiz['score']],
                'questions_attempted': quiz['correct_answers'] + quiz['incorrect_answers'],
                'correct_answers': quiz['correct_answers'],
                'incorrect_answers': quiz['incorrect_answers'],
                'mistake_correction_rate': (quiz['mistakes_corrected'] / quiz['initial_mistake_count'] * 100)
                if quiz['initial_mistake_count'] > 0 else 100
            }
        else:
            topic_accuracy[topic]['accuracies'].append(accuracy)
            topic_accuracy[topic]['scores'].append(quiz['score'])

    # Process current quiz topic
    if current_topic not in topic_accuracy:
        topic_accuracy[current_topic] = {
            'accuracies': [current_accuracy],
            'scores': [current_quiz['score']],
            'questions_attempted': current_quiz['correct_answers'] + current_quiz['incorrect_answers'],
            'correct_answers': current_quiz['correct_answers'],
            'incorrect_answers': current_quiz['incorrect_answers'],
            'mistake_correction_rate': (current_quiz['mistakes_corrected'] / current_quiz['initial_mistake_count'] * 100)
            if current_quiz['initial_mistake_count'] > 0 else 100
        }
    else:
        topic_accuracy[current_topic]['accuracies'].append(current_accuracy)
        topic_accuracy[current_topic]['scores'].append(current_quiz['score'])

    # Calculate final metrics for each topic
    for topic, metrics in topic_accuracy.items():
        metrics['average_accuracy'] = sum(metrics['accuracies']) / len(metrics['accuracies'])
        metrics['current_accuracy'] = metrics['accuracies'][-1]
        metrics['previous_accuracy'] = metrics['accuracies'][-2] if len(metrics['accuracies']) > 1 else metrics['accuracies'][-1]
        metrics['improvement'] = metrics['current_accuracy'] - metrics['previous_accuracy']
        metrics['average_score'] = sum(metrics['scores']) / len(metrics['scores'])
        metrics['std_dev_accuracy'] = np.std(metrics['accuracies']) if len(metrics['accuracies']) > 1 else 0.0
        metrics['std_dev_score'] = np.std(metrics['scores']) if len(metrics['scores']) > 1 else 0.0

    # Calculate overall metrics
    overall_stats = {
        'current_score': current_quiz['score'],  # Use the correct current score (32)
        'accuracy': float(current_quiz['accuracy'].strip(' %')),  # Use the correct accuracy (80.0%)
        'speed_rating': float(current_quiz['speed']),  # Use the correct speed (100.0)
        'percentile': current_quiz['better_than'],  # Use the correct percentile (24)
        'trophy_level': current_quiz['trophy_level'],  # Use the correct trophy level (2)
        'improvement_rate': (current_quiz['mistakes_corrected'] / current_quiz['initial_mistake_count'] * 100)
        if current_quiz['initial_mistake_count'] > 0 else 100  # Correctly calculate improvement rate (75.0%)
    }

    return {
        "Quiz Performance Analysis": {
            "Topic Accuracy": topic_accuracy,
            "Overall Performance": overall_stats,
            "Detailed Metrics": {
                "Current Quiz": {
                    "Score": current_quiz['score'],
                    "Accuracy": current_accuracy,
                    "Speed": current_quiz['speed'],
                    "Final Score": float(current_quiz['final_score']),
                    "Negative Score": float(current_quiz['negative_score']),
                    "Better Than": current_quiz['better_than'],
                    "Total Questions": current_quiz['total_questions'],
                    "Mistakes Corrected": current_quiz['mistakes_corrected'],
                    "Initial Mistake Count": current_quiz['initial_mistake_count'],
                    "Topic": current_topic
                },
                "Historical Comparison": calculate_historical_comparison(current_quiz, historical_quizzes),
                "Progress Indicators": {
                    "Accuracy Trend": calculate_accuracy_trend(current_quiz, historical_quizzes),
                    "Score Trend": calculate_score_trend(current_quiz, historical_quizzes),
                    "Improvement Trend": calculate_improvement_trend(current_quiz, historical_quizzes)
                }
            }
        },
        "Insights": generate_insights(topic_accuracy, overall_stats),
        "Recommendations": generate_recommendations(topic_accuracy, overall_stats),
        "Student Persona": analyze_student_persona(topic_accuracy, overall_stats)
    }

def calculate_historical_comparison(current: Dict, historical: List[Dict]) -> Dict:
    """Compare current quiz performance with historical data."""
    if not historical:
        return {"comparison_available": False}

    current_score = current['score']
    current_accuracy = float(current['accuracy'].strip(' %'))
    current_percentile = current['better_than']

    historical_scores = [h['score'] for h in historical]
    historical_accuracies = [float(h['accuracy'].strip(' %')) for h in historical]
    historical_percentiles = [h['better_than'] for h in historical]

    avg_historical_score = sum(historical_scores) / len(historical_scores)
    avg_historical_accuracy = sum(historical_accuracies) / len(historical_accuracies)
    avg_historical_percentile = sum(historical_percentiles) / len(historical_percentiles)

    # Include recent scores and accuracies for trend analysis
    recent_scores = [str(h['score']) for h in historical[-3:]]  # Last 3 scores
    recent_accuracies = [f"{float(h['accuracy'].strip(' %')):.1f}%" for h in historical[-3:]]  # Last 3 accuracies

    # Explain percentile trend
    percentile_trend_explanation = (
        "The percentile is declining because the student's performance is improving slower than their peers, "
        "or the overall performance of the group has improved significantly."
    )

    return {
        "comparison_available": True,
        "score_change": current_score - avg_historical_score,
        "accuracy_change": current_accuracy - avg_historical_accuracy,
        "percentile_change": current_percentile - avg_historical_percentile,
        "baseline_score": avg_historical_score,
        "baseline_accuracy": avg_historical_accuracy,
        "baseline_percentile": avg_historical_percentile,
        "recent_scores": recent_scores,
        "recent_accuracies": recent_accuracies,
        "trend": {
            "score": "improving" if current_score > avg_historical_score else "declining",
            "accuracy": "improving" if current_accuracy > avg_historical_accuracy else "declining",
            "percentile": "improving" if current_percentile > avg_historical_percentile else "declining",
            "percentile_explanation": percentile_trend_explanation if current_percentile < avg_historical_percentile else None
        }
    }

def calculate_accuracy_trend(current: Dict, historical: List[Dict]) -> List[float]:
    """Calculate accuracy trend over time."""
    return [float(q['accuracy'].strip(' %')) for q in historical + [current]]

def calculate_score_trend(current: Dict, historical: List[Dict]) -> List[int]:
    """Calculate score trend over time."""
    return [q['score'] for q in historical + [current]]

def calculate_improvement_trend(current: Dict, historical: List[Dict]) -> List[float]:
    """Calculate improvement trend over time."""
    return [(q['mistakes_corrected'] / q['initial_mistake_count'] * 100)
            if q['initial_mistake_count'] > 0 else 100
            for q in historical + [current]]

def generate_insights(topic_accuracy: Dict, overall_stats: Dict) -> Dict:
    """Generate insights based on performance analysis."""
    insights = {
        "Weak Areas": [],
        "Improvement Trends": [],
        "Performance Gaps": []
    }

    for topic, metrics in topic_accuracy.items():
        if metrics['average_accuracy'] is not None and metrics['average_accuracy'] < 80:
            insights["Weak Areas"].append(topic)

    if overall_stats['improvement_rate'] > 50:
        insights["Improvement Trends"].append("Improvement in mistake correction rate")

    if overall_stats['accuracy'] is not None and overall_stats['accuracy'] < 80:
        insights["Performance Gaps"].append("Low accuracy")

    return insights

def generate_recommendations(topic_accuracy: Dict, overall_stats: Dict) -> List[str]:
    """Generate specific and actionable recommendations based on performance."""
    recommendations = []

    # Group topics by performance level
    weak_topics = []
    moderate_topics = []
    strong_topics = []

    for topic, metrics in topic_accuracy.items():
        if metrics['current_accuracy'] < 60:
            weak_topics.append((topic, metrics['current_accuracy']))
        elif metrics['current_accuracy'] < 80:
            moderate_topics.append((topic, metrics['current_accuracy']))
        else:
            strong_topics.append((topic, metrics['current_accuracy']))

    # Sort topics by accuracy
    weak_topics.sort(key=lambda x: x[1])
    moderate_topics.sort(key=lambda x: x[1])
    strong_topics.sort(key=lambda x: x[1], reverse=True)

    # Generate specific recommendations
    if strong_topics:
        top_performer = strong_topics[0]
        recommendations.append(
            f"Maintain excellence in {top_performer[0]} ({top_performer[1]:.1f}%) by regularly practicing advanced questions"
        )

    if weak_topics:
        worst_performers = weak_topics[:2]  # Focus on top 2 weakest topics
        for topic, accuracy in worst_performers:
            recommendations.append(
                f"Prioritize {topic} (current accuracy: {accuracy:.1f}%) - Focus on fundamental concepts and practice basic questions"
            )

    if moderate_topics:
        for topic, accuracy in moderate_topics:
            recommendations.append(
                f"Strengthen {topic} (current accuracy: {accuracy:.1f}%) - Practice medium difficulty questions and review weak areas"
            )

    return recommendations

def analyze_student_persona(topic_accuracy: Dict, overall_stats: Dict) -> Dict:
    """Analyze student persona based on actual performance data."""
    # Define thresholds
    STRENGTH_THRESHOLD = 80
    WEAKNESS_THRESHOLD = 60

    strengths = []
    weaknesses = []

    # Analyze topic performance
    for topic, metrics in topic_accuracy.items():
        if metrics['average_accuracy'] >= STRENGTH_THRESHOLD:
            strengths.append(f"{topic} ({metrics['average_accuracy']:.1f}%)")
        elif metrics['average_accuracy'] < WEAKNESS_THRESHOLD:
            weaknesses.append(f"{topic} ({metrics['average_accuracy']:.1f}%)")

    # Determine learning style based on speed and accuracy
    learning_style = determine_learning_style(overall_stats['speed_rating'], overall_stats['accuracy'])

    # Determine motivation level based on improvement rate and trophy level
    motivation = determine_motivation_level(overall_stats['improvement_rate'], overall_stats['trophy_level'])

    return {
        "Strengths": strengths,
        "Weaknesses": weaknesses,
        "Learning Style": learning_style,
        "Motivation": motivation,
        "Performance Pattern": analyze_performance_pattern(topic_accuracy)
    }

def analyze_performance_pattern(topic_accuracy: Dict) -> str:
    """Analyze overall performance pattern."""
    topic_improvements = [metrics['improvement'] for metrics in topic_accuracy.values()]
    avg_improvement = sum(topic_improvements) / len(topic_improvements) if topic_improvements else 0

    if avg_improvement > 10:
        return "Strong improvement across topics"
    elif avg_improvement > 0:
        return "Steady improvement"
    else:
        return "Needs focused improvement strategy"

def determine_learning_style(speed_rating: float, accuracy: float) -> str:
    """Determine learning style based on speed and accuracy metrics."""
    if speed_rating >= 80 and accuracy >= 80:
        return "Fast and accurate learner"
    elif speed_rating >= 80:
        return "Quick but needs to focus on accuracy"
    elif accuracy >= 80:
        return "Methodical and accurate learner"
    else:
        return "Developing learning pattern"

def determine_motivation_level(improvement_rate: float, trophy_level: int) -> str:
    """Determine motivation level based on improvement and achievements."""
    if improvement_rate >= 80 and trophy_level >= 3:
        return "Highly motivated"
    elif improvement_rate >= 60 or trophy_level >= 2:
        return "Moderately motivated"
    else:
        return "Needs motivation boost"

def visualize_performance(current: Dict, historical: List[Dict]) -> None:
    """Visualize the performance trends using matplotlib."""
    accuracy_trend = calculate_accuracy_trend(current, historical)
    score_trend = calculate_score_trend(current, historical)

    plt.plot(accuracy_trend)
    plt.xlabel('Quiz')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Trend')
    plt.show()

    plt.plot(score_trend)
    plt.xlabel('Quiz')
    plt.ylabel('Score')
    plt.title('Score Trend')
    plt.show()

def display_enhanced_performance(result, historical_quiz_data):
    """Display the enhanced performance report using rich formatting."""
    console.rule("[bold blue]📊 Quiz Performance Analysis 📊[/bold blue]")

    topic_table = Table(show_header=True, header_style="bold magenta")
    topic_table.add_column("Topic", style="dim", width=30)
    topic_table.add_column("Current Accuracy", justify="right")
    topic_table.add_column("Avg Score", justify="right")
    topic_table.add_column("Questions", justify="right")
    topic_table.add_column("Correct", justify="right")
    topic_table.add_column("Incorrect", justify="right")
    topic_table.add_column("Improvement", justify="right")

    for topic, metrics in result["Quiz Performance Analysis"]["Topic Accuracy"].items():
        current_accuracy = metrics['current_accuracy']
        avg_score = metrics['average_score']
        improvement = metrics['improvement']

        topic_table.add_row(
            topic,
            f"[{'green' if current_accuracy >= 80 else 'yellow' if current_accuracy >= 60 else 'red'}]{current_accuracy:.1f}%[/]",
            f"{avg_score:.1f}",
            str(metrics['questions_attempted']),
            f"[green]{metrics['correct_answers']}[/]",
            f"[red]{metrics['incorrect_answers']}[/]",
            f"[{'green' if improvement > 0 else 'red'}]{improvement:+.1f}%[/]"
        )
    console.print(topic_table)

    # Update trend analysis panel
    historical = result["Quiz Performance Analysis"]["Detailed Metrics"]["Historical Comparison"]
    if historical["comparison_available"]:
        trend_details = Text.assemble(
            ("Score Change: ", "bold"),
            (f"{historical['score_change']:+.1f} ", "green" if historical['score_change'] >= 0 else "red"),
            
            

            ("\nAccuracy Change: ", "bold"),
            (f"{historical['accuracy_change']:+.1f}% ", "green" if historical['accuracy_change'] >= 0 else "red"),
           
            

            ("\nPercentile Change: ", "bold"),
            (f"{historical['percentile_change']:+.1f} ", "green" if historical['percentile_change'] >= 0 else "red"),
            

            ("\nPerformance Trends:\n", "bold"),
            (f"• Score: {historical['trend']['score']} (Last 3 quizzes: {', '.join(historical['recent_scores'])})\n",
             "green" if historical['trend']['score'] == 'improving' else "red"),
            (f"• Accuracy: {historical['trend']['accuracy']} (Last 3 accuracies: {', '.join(historical['recent_accuracies'])})\n",
             "green" if historical['trend']['accuracy'] == 'improving' else "red"),
            (f"• Percentile: {historical['trend']['percentile']}",
             "green" if historical['trend']['percentile'] == 'improving' else "red")
        )
        if historical['trend']['percentile_explanation']:
            trend_details.append(f"\n{historical['trend']['percentile_explanation']}")

        trends_panel = Panel(trend_details, title="[bold]Trend Analysis[/bold]", border_style="blue")
    else:
        trends_panel = Panel("Not enough historical data for comparison", title="[bold]Trend Analysis[/bold]", border_style="blue")
    console.print(trends_panel)

    persona = result["Student Persona"]
    persona_panel = Panel(
        Text.assemble(
            ("Learning Style: ", "bold"),
            (f"{persona['Learning Style']}\n", "cyan"),
            ("Motivation Level: ", "bold"),
            (f"{persona['Motivation']}\n", "cyan"),
            ("Performance Pattern: ", "bold"),
            (f"{persona['Performance Pattern']}\n\n", "cyan"),
            ("Strengths: ", "bold"),
            (f"{', '.join(persona['Strengths']) or 'Developing'}\n", "green"),
            ("Areas for Improvement: ", "bold"),
            (f"{', '.join(persona['Weaknesses']) or 'None identified'}", "yellow")
        ),
        title="[bold]Student Profile[/bold]",
        border_style="cyan"
    )
    console.print(persona_panel)

    insights = result["Insights"]
    console.print("\n[bold]Key Insights:[/bold]")
    if insights["Improvement Trends"]:
        console.print("[green]✓ " + "\n✓ ".join(insights["Improvement Trends"]))
    if insights["Performance Gaps"]:
        console.print("[yellow]⚠ " + "\n⚠ ".join(insights["Performance Gaps"]))

    recommendations = result["Recommendations"]
    console.print("\n[bold]Recommended Actions:[/bold]")
    for i, rec in enumerate(recommendations, 1):
        console.print(f"[cyan]{i}. {rec}")

    overall = result["Quiz Performance Analysis"]["Overall Performance"]
    summary_panel = Panel(
        Text.assemble(
            ("Current Score: ", "bold"),
            (f"{overall['current_score']}\n", "green"),
            ("Accuracy: ", "bold"),
            (f"{overall['accuracy']:.1f}%\n", "green"),
            ("Speed Rating: ", "bold"),
            (f"{overall['speed_rating']:.1f}\n", "cyan"),
            ("Percentile: ", "bold"),
            (f"{overall['percentile']}\n", "yellow"),
            ("Trophy Level: ", "bold"),
            ("🏆 " * overall['trophy_level'], "gold1"),
            ("\nImprovement Rate: ", "bold"),
            (f"{overall['improvement_rate']:.1f}%", "cyan")
        ),
        title="[bold]Overall Performance[/bold]",
        border_style="green"
    )
    console.print(summary_panel)

def main():
    """Main function to fetch data, analyze performance, and display results."""
    current_quiz_data = get_quiz_performance_data(quiz_performance_api_key)
    historical_quiz_data = get_historical_performance_data(historical_performance_api_key)

    result = analyze_quiz_performance(current_quiz_data, historical_quiz_data)

    display_enhanced_performance(result, historical_quiz_data)

    visualize_performance(current_quiz_data, historical_quiz_data)

if __name__ == "__main__":
    main()