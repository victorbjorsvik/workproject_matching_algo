import sqlite3

# Connect to SQLite database
conn = sqlite3.connect('skills.db')
cursor = conn.cursor()

courses = {
    "Python": {
        "Source": "Harvard",
        "Course": "Computer Science for Python Programming",
        "Duration in hours": 204,
        "Cost of specific course": 430
    },
    "R": {
        "Source": "DataCamp",
        "Course": "Associate Data Scientist in R",
        "Duration in hours": 88,
        "Cost of specific course": 335
    },
    "AWS": {
        "Source": "Amazon",
        "Course": "Data Scientist Track",
        "Duration in hours": 166,
        "Cost of specific course": 449
    },
    "C++": {
        "Source": "OpenEDG",
        "Course": "C++ Essentials – Part 1/2 and Advanced",
        "Duration in hours": 126,
        "Cost of specific course": 0
    },
    "Apache Hadoop": {
        "Source": "Project Pro",
        "Course": "Apache Hadoop Roadmap for Data Science and Analytics",
        "Duration in hours": 60,
        "Cost of specific course": 360
    },
    "Apache Spark": {
        "Source": "Project Pro",
        "Course": "Apache Spark Learning path",
        "Duration in hours": 40,
        "Cost of specific course": 360
    },
    "Microsoft Azure": {
        "Source": "Microsoft",
        "Course": "Learning path Beginner + Intermediate + Advanced",
        "Duration in hours": 144,
        "Cost of specific course": 0
    },
    "Microsoft Excel": {
        "Source": "Coursera",
        "Course": "Data Analysis with Excel Part 1 and Part 2",
        "Duration in hours": 37,
        "Cost of specific course": 399
    },
    "Power BI": {
        "Source": "Coursera",
        "Course": "Microsoft Power BI Data Analyst Certificate Course",
        "Duration in hours": 189,
        "Cost of specific course": 399
    },
    "Oracle Java": {
        "Source": "Udemy",
        "Course": "Masterclass",
        "Duration in hours": 135,
        "Cost of specific course": 120
    },
    "SQL": {
        "Source": "Coursera",
        "Course": "Learn SQL Basics for Data Science Specialization",
        "Duration in hours": 80,
        "Cost of specific course": 399
    },
    "Tableau": {
        "Source": "Tableau",
        "Course": "Data Scientist Learning Path",
        "Duration in hours": 130,
        "Cost of specific course": 1380
    },
    "NoSQL": {
        "Source": "Class central",
        "Course": "NoSQL Databases",
        "Duration in hours": 54,
        "Cost of specific course": 0
    },
    "Git": {
        "Source": "The Linux Foundation",
        "Course": "Open Source Software Development Linux and Git Specialization",
        "Duration in hours": 48,
        "Cost of specific course": 399
    },
    "SAS": {
        "Source": "SAS",
        "Course": "Advanced Analytics Professional",
        "Duration in hours": 118,
        "Cost of specific course": 875
    },
    "Scala": {
        "Source": "Coursera",
        "Course": "Functional Programming in Scala Specialization",
        "Duration in hours": 182,
        "Cost of specific course": 399
    }
}




for k, v in courses.items():
    li = [x for _, x in v.items()]
    cursor.execute('''INSERT INTO required_skills (skill, source, course, duration, cost) VALUES (?, ?, ?, ?, ?)''', (k, li[0], li[1], li[2], li[3]))


# Commit changes and close connection
conn.commit()
conn.close()