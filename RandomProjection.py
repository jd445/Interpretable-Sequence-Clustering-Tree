import random
import threading


class RandomProjection:
    def __init__(self, seqs, max_feature_num, max_feature_len):
        self.seqs = seqs
        self.max_feature_num = max_feature_num
        self.max_feature_len = max_feature_len
        self.itemset = self.generate_itemset(seqs)  # Generate the itemset

    def generate_itemset(self, seqs):
        """
        Generate the itemset from the sequences.
        """
        # Flatten all sequences and extract unique items
        itemset = set(item for sublist in seqs for item in sublist)
        return list(itemset)

    def generate_features(self):
        """
        Generate random features based on the itemset.
        """
        features = set()  # To store unique features
        for _ in range(self.max_feature_num):
            feature = []
            # Randomly determine the length of the feature
            feature_length = random.randint(1, self.max_feature_len)

            for _ in range(feature_length):
                # Randomly select an item from the itemset
                random_item = random.choice(self.itemset)
                feature.append(random_item)

            # Add the feature to the set
            features.add(tuple(feature))  # Use tuple to make it hashable

        print(f"Feature Number = {len(features)}")
        return list(features)

    @staticmethod
    def longest_common_substring_similarity(a, b):
        """
        Calculate the longest common substring similarity between two sequences.
        """
        if len(a) < len(b):
            a, b = b, a  # Swap to ensure a is always larger or equal in size

        dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        length = 0

        for i in range(1, len(a) + 1):
            for j in range(1, len(b) + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    length = max(length, dp[i][j])
                else:
                    dp[i][j] = 0

        return float(length)

    def generate_feature_vector(self, features):
        """
        Generate the feature vector for all sequences using multithreading.
        """
        num_of_threads = 8  # Number of threads
        num_of_sequences = len(self.seqs)
        feature_vector = [[0] * len(features) for _ in range(num_of_sequences)]
        threads = []

        # Split the sequences into chunks for each thread
        sequences_per_thread = num_of_sequences // num_of_threads
        start = 0

        for i in range(num_of_threads):
            end = start + sequences_per_thread
            if i == num_of_threads - 1:  # Handle the remaining sequences in the last thread
                end = num_of_sequences

            # Create and start a thread
            thread = threading.Thread(
                target=self.generate_feature_vector_thread,
                args=(start, end, features, feature_vector),
            )
            threads.append(thread)
            thread.start()
            start = end

        # Wait for all threads to finish
        for thread in threads:
            thread.join()

        return feature_vector

    def generate_feature_vector_thread(self, start, end, features, feature_vector):
        """
        Threaded function to compute feature vector.
        """
        for i in range(start, end):
            for j, feature in enumerate(features):
                feature_vector[i][j] = self.longest_common_substring_similarity(self.seqs[i], feature)

'''
# Example usage
a = ['0', '1', '2', '3', '4', '5', '8', '5', '1', '5', '7', '5', '1', '8', '5', '1', '5', '1', '5', '1', '6', '4', '5']
b = ['0', '1', '2', '3', '4', '5', '1', '5', '1', '6', '4', '5', 'aa']

test_data = [a, b]

# Create an instance of RandomProjection
rp = RandomProjection(test_data, max_feature_num=10, max_feature_len=5)

# Generate features
features = rp.generate_features()

# Generate feature vector using multithreading
feature_vector = rp.generate_feature_vector(features)
'''