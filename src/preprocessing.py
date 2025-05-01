# Download the dataset
data_path = download_dataset()

if not data_path:
    raise ValueError("Failed to download the dataset. Please check the URL.")

# Load the dataset
print("\n1. LOADING DATASET")
print("-------------------")

# Define column names based on the UCI documentation
column_names = [f'A{i}' for i in range(1, 16)] + ['class']

# Load the data
data = pd.read_csv(data_path, header=None, names=column_names, na_values='?')

# Convert class labels to binary (1 for '+' approved, 0 for '-' denied)
data['class'] = data['class'].map({'+': 1, '-': 0})

print(f"Dataset shape: {data.shape}")
print("\nData preview:")
print(data.head())

print("\nMissing values per column:")
print(data.isnull().sum())

print("\nClass distribution:")
class_counts = data['class'].value_counts()
print(class_counts)

# Plot class distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='class', data=data)
plt.title('Credit Approval Distribution')
plt.xlabel('Approved (1) / Denied (0)')
plt.ylabel('Count')
plt.savefig(os.path.join(output_dir, 'eda', 'class_distribution.png'))
plt.show()

# Identify categorical and numerical columns
cat_cols = data.select_dtypes(include=['object']).columns.tolist()
num_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Remove target variable from features list
if 'class' in num_cols:
    num_cols.remove('class')

print("\nCategorical columns:", cat_cols)
print("Numerical columns:", num_cols)