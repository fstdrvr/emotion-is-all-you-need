from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago 
from datetime import timedelta

from data_scraping import scrape_youtube_comments
from nlp_modeling import load_bert_model, load_spacy_model  # Assumes models are loaded once 
from sentiment_analysis import calculate_sentiment, aggregate_video_sentiment
from comment_summarization import summarize_comments_gpt4
from googleapiclient.discovery import build

# Get YouTube video urls from home page
def get_video_urls(**kwargs):
    api_key = "YOUR_YOUTUBE_API_KEY"  
    youtube = build("youtube", "v3", developerKey=api_key)

    request = youtube.videos().list(
        part="id",
        chart="mostPopular",
        regionCode="US"  # Adjust region if needed
    )
    response = request.execute()

    video_urls = [f"https://www.youtube.com/watch?v={item['id']}"] for item in response['items']]
    return video_urls

# Default settings applied to all tasks
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,  
    'email_on_failure': False,  
    'email_on_retry': False, 
    'retries': 1,
    'retry_delay': timedelta(minutes=5) 
}

with DAG(
    dag_id='youtube_sentiment_analysis',
    default_args=default_args,
    description='Analyzes YouTube video sentiment based on comments',
    schedule_interval=timedelta(days=1),  # Adjust your run frequency
    start_date=days_ago(2),  
    tags=['youtube', 'sentiment', 'nlp'], 
) as dag:

    get_videos_task = PythonOperator(
        task_id='get_video_urls',
        python_callable=get_video_urls,
    )

    scrape_comments_task = PythonOperator(
        task_id='scrape_comments',
        python_callable=scrape_youtube_comments,  
        op_kwargs={'video_url': "{{ task_instance.xcom_pull(task_ids='get_video_urls') }}" },  # Passes URLs dynamically
    )

    sentiment_analysis_task = PythonOperator(
        task_id='analyze_sentiment',
        python_callable=aggregate_video_sentiment,  
        op_kwargs={'comments': "{{ task_instance.xcom_pull(task_ids='scrape_comments') }}" },  
    )

    summarize_comments_task = PythonOperator(
        task_id='summarize_comments',
        python_callable=summarize_comments_gpt4,  
        op_kwargs={'comments': "{{ task_instance.xcom_pull(task_ids='scrape_comments') }}" },  
    )

    get_videos_task >> scrape_comments_task >> sentiment_analysis_task >> summarize_comments_task