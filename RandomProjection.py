import random
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


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
        Generate the feature vector for all sequences using multiprocessing.
        """
        num_of_processes = min(cpu_count(), len(self.seqs))  # Number of processes
        sequence_chunks = self.split_sequences(num_of_processes)  # Split sequences into chunks

        # Prepare arguments for each process
        args = [(chunk, features) for chunk in sequence_chunks]

        # Use a multiprocessing pool to process each chunk
        with Pool(num_of_processes) as pool:
            # Use tqdm to track progress
            with tqdm(total=len(self.seqs), desc="Feature Vector Calculation") as pbar:
                results = []
                for result in pool.imap(self.generate_feature_vector_chunk, args):
                    results.append(result)
                    pbar.update(len(result))  # Update progress bar based on chunk size

        # Merge results from all processes
        feature_vector = [vector for chunk in results for vector in chunk]
        return feature_vector

    @staticmethod
    def generate_feature_vector_chunk(args):
        """
        Worker function to compute feature vector for a chunk of sequences.
        """
        chunk, features = args
        feature_vector = []

        for seq in chunk:
            vector = []
            for feature in features:
                similarity = RandomProjection.longest_common_substring_similarity(seq, feature)
                vector.append(similarity)
            feature_vector.append(vector)

        return feature_vector

    def split_sequences(self, num_of_chunks):
        """
        Split self.seqs into num_of_chunks parts.
        """
        chunk_size = len(self.seqs) // num_of_chunks
        chunks = [self.seqs[i * chunk_size:(i + 1) * chunk_size] for i in range(num_of_chunks - 1)]
        chunks.append(self.seqs[(num_of_chunks - 1) * chunk_size:])  # Add remaining sequences to the last chunk
        return chunks


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