print("-"*30 + "STEP 1 :creating data" + "-"*30)

users = ["user1","user2","user3","user4","user5","user6","user7"]
posts = ["post1","post2","post3","post4","post5","post6","post7","post8","post9"]
interaction = {
    "user1" : ["post1","post3","post9","post6"],
    "user2" : ["post4","post2","post9","post8"],
    "user3" : ["post5","post7","post8","post6"],
    "user4" : ["post9","post6","post8","post1"],
    "user5" : ["post1","post5","post6","post9"],
    "user6" : ["post2","post3","post5","post6"],
    "user7" : ["post3","post9","post2","post8"]
    }

print("📊The availbale users :")
for user in users:
    print(f"-{user}")
print("📷the availbale posts : ")    
for post in posts:
   print(f"-{post}")
print("💝the intaraction : ")

for x,y in interaction.items():
    print(f"{x} . liked : {' ,'.join(y)}")

print("-"*30 + "STEP 2 :finding similar users" + "-"*30)

def similar(this_user ):
    this_user_likes = set(interaction[this_user])
    similaries = {}
    for other_user,other_like in interaction.items():
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
