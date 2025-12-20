import mysql.connector

print("-"*30 + "STEP 1 :creating data" + "-"*30)

users = ["user1","user2","user3","user4","user5","user6","user7"]
posts = ["post1","post2","post3","post4","post5","post6","post7","post8","post9"]
interactions = {
    "user1" : ["post1","post3","post9","post6"],
    "user2" : ["post4","post2","post9","post8"],
    "user3" : ["post5","post7","post8","post6"],
    "user4" : ["post9","post6","post8","post1"],
    "user5" : ["post1","post5","post6","post9"],
    "user6" : ["post2","post3","post5","post6"],
    "user7" : ["post3","post9","post2","post8"]
    }




print("Data inserted successfully!")

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
    print(sug_post)
suggest(similar_users)

try:
    connection = mysql.connector.connect(
        host='https://192.168.1.101',  # Your main PC's IP address
        user='remote_user',
        password='2929',
        database='instagrame_data',
        port=3306
    )
    
    if connection.is_connected():
        print("✓ Successfully connected from remote PC!")
        
        # Test query
        cursor = connection.cursor()
        cursor.execute("SELECT DATABASE()")
        db_name = cursor.fetchone()
        print(f"Connected to database: {db_name[0]}")
        
    connection.close()
    
except Exception as e:
    print(f"Connection failed: {e}")
