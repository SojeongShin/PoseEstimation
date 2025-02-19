#!/usr/bin/env python3

from arcface import ArcFace

def main():
    # Create an ArcFace instance
    face_rec = ArcFace.ArcFace()

    # Calculate embedding for the first image
    emb1 = face_rec.calc_emb("sj-cap.jpeg")
    print("Embedding for first image (test.jpg):")
    print(emb1)

    # Calculate embedding for the second image
    emb2 = face_rec.calc_emb("Identification/sj.jpg")
    print("\nEmbedding for second image (test2.jpg):")
    print(emb2)

    # Calculate the distance between the two embeddings
    distance = face_rec.get_distance_embeddings(emb1, emb2)
    print(f"\nDistance between the two embeddings: {distance:.5f}")

if __name__ == "__main__":
    main()
