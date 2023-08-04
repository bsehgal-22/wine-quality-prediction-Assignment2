
from pyspark.sql import SparkSession

def main():
    # Create a SparkSession
    spark = SparkSession.builder.appName("MatrixMultiplication").getOrCreate()

    # Sample matrices
    matrix_a = spark.sparkContext.parallelize([(1, 2), (3, 4)])
    matrix_b = spark.sparkContext.parallelize([(1, 2), (3, 4)])

    # Matrix multiplication
    def multiply(row_col_pair):
        row, col = row_col_pair
        return row[0] * col[1]

    matrix_c = matrix_a.cartesian(matrix_b).map(multiply).reduce(lambda a, b: a + b)

    print(">>>>>>", matrix_c)
    print(">>>>>>Matrix C (Result of Multiplication):", matrix_c)
    print(">>>>>>", matrix_c)


    # Stop the SparkSession
    spark.stop()

if __name__ == "__main__":
    main()
