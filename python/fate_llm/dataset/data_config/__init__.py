import os
# absolute path to current directory
parent_dir = os.path.dirname(os.path.realpath(__file__))

DATA_CONFIG_TEMPLATE = {"ag_news": os.path.join(parent_dir, "default_ag_news.yaml"),
                        "yelp_review": os.path.join(parent_dir, "default_yelp_review.yaml"),}