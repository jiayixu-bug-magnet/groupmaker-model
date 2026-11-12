import pandas as pd
from sklearn.cluster import KMeans

SKILL_COLUMNS = [
    "skill_programming",
    "skill_data_stats",
    "skill_frontend_design",
    "skill_business_pitch",
]
K = 3


def main():
    students = pd.read_csv("class_dataset.csv")
    skills = students[SKILL_COLUMNS]

    model = KMeans(n_clusters=K, n_init=10, random_state=0)
    students["cluster"] = model.fit_predict(skills)

    print(f"students: {len(students)}")
    for cluster_id, centroid in enumerate(model.cluster_centers_):
        count = int((students["cluster"] == cluster_id).sum())
        averages = ", ".join(
            f"{column}={value:.2f}" for column, value in zip(SKILL_COLUMNS, centroid)
        )
        print(f"cluster {cluster_id}: n={count} {averages}")

    print()
    for row in students.itertuples(index=False):
        print(f"{row.netid},{row.cluster}")


if __name__ == "__main__":
    main()
