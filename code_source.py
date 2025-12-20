import mysql.connector
import pandas as pd

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="2929",     
    database="instagrame_data"
)

cursor = db.cursor()

def get_all_users():
    """Get all users from the database"""
    cursor.execute("SELECT user_id, username FROM users")
    users = cursor.fetchall()
    return users

def get_all_posts():
    """Get all posts from the database"""
    cursor.execute("SELECT post_id, post_name FROM posts")
    posts = cursor.fetchall()
    return posts

def get_user_interactions(user_id=None):
    """
    Get interactions from database
    If user_id is provided, get only that user's interactions
    """
    if user_id:
        cursor.execute("""
            SELECT u.username, p.post_name 
            FROM interactions i
            JOIN users u ON i.user_id = u.user_id
            JOIN posts p ON i.post_id = p.post_id
            WHERE u.user_id = %s
        """, (user_id,))
    else:
        cursor.execute("""
            SELECT u.username, p.post_name 
            FROM interactions i
            JOIN users u ON i.user_id = u.user_id
            JOIN posts p ON i.post_id = p.post_id
            ORDER BY u.username
        """)
    return cursor.fetchall()

def get_interaction_matrix():
    """Get interactions data as a dictionary similar to your original 'interactions' variable"""
    cursor.execute("""
        SELECT u.username, GROUP_CONCAT(p.post_name) as liked_posts
        FROM interactions i
        JOIN users u ON i.user_id = u.user_id
        JOIN posts p ON i.post_id = p.post_id
        GROUP BY u.username
    """)
    interactions = {}
    for username, liked_posts in cursor.fetchall():
        interactions[username] = liked_posts.split(',')
    return interactions

users = get_all_users()
posts = get_all_posts()
interactions = get_interaction_matrix()

print("📊The availbale users :")
for user in users:
    print(f"-{user}")
print("📷the availbale posts : ")    
for post in posts:
   print(f"-{post}")
print("💝the intaraction : ")

for x,y in interactions.items():
    print(f"{x} . liked : {' ,'.join(y)}")

print("-"*30 + "STEP 2 :finding similar users" + "-"*30)

def similar(this_user ):
    this_user_likes = set(interactions[this_user])
    similaries = {}
    for other_user,other_like in interactions.items():
        if this_user != other_user:
            other_like = set(other_like)
            common_post = this_user_likes.intersection(other_like)
            similarity_score = len(common_post)
            if similarity_score>0:
                similaries[other_user] = {
                    "score" : similarity_score,
                    "common_posts" : list(common_post)
                    }
    return similaries
similar_user1 = similar("user1")

for user,info in similar_user1.items():
    print(f"  {user}:")
    print(f"    Similarity score: {info['score']}")
    print(f"    Both liked: {', '.join(info['common_posts'])}")

def get_sim_users(similar_user1):
    max_ = max(item['score'] for item in similar_user1.values())
    max_users = [user for user, item in similar_user1.items() 
                 if item['score'] == max_]
    return max_users
similar_users = get_sim_users(similar_user1)

def suggest(similar_users):
    sug_post = []
    for user in similar_users:
        if len(similar_users) > 1:
            for user_,inf in interactions.items():
                if user == user_:
                    sug_post.append(inf)
#convert list to det an s this set to list and if not exist in
    print(sug_post)
suggest(similar_users)
            