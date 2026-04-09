# kafka/topics.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

KAFKA_BOOTSTRAP = "localhost:9092"

# Topic names
TOPIC_PRICES    = "market.prices"
TOPIC_NEWS      = "market.news"
TOPIC_MACRO     = "market.macro"
TOPIC_FEATURES  = "market.features"
TOPIC_SENTIMENT = "market.sentiment"
TOPIC_SIGNALS   = "market.signals"

def create_topics():
    """Create all Kafka topics if they don't exist."""
    from kafka.admin import KafkaAdminClient, NewTopic
    admin = KafkaAdminClient(bootstrap_servers=KAFKA_BOOTSTRAP)
    existing = admin.list_topics()

    topics_to_create = []
    for name in [TOPIC_PRICES, TOPIC_NEWS, TOPIC_MACRO,
                 TOPIC_FEATURES, TOPIC_SENTIMENT, TOPIC_SIGNALS]:
        if name not in existing:
            topics_to_create.append(
                NewTopic(name=name, num_partitions=1,
                         replication_factor=1))

    if topics_to_create:
        admin.create_topics(topics_to_create)
        print(f"Created topics: {[t.name for t in topics_to_create]}")
    else:
        print("All topics already exist.")
    admin.close()

if __name__ == "__main__":
    create_topics()
