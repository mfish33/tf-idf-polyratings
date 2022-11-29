import org.apache.spark.{SparkConf, SparkContext}
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, count, countDistinct, explode, pow, size, split, sqrt, sum, udf}

object fastTfidf {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val sparkSession = SparkSession
      .builder()
      .config("spark.master", "local")
      .appName("myApp")
      .getOrCreate()

    import sparkSession.implicits._

    val ratings = sparkSession
      .read
      .json(sparkSession.sparkContext.wholeTextFiles("./ratings.json").values)
      .select($"id", $"professor", $"rating".as("document"))
      .cache()

    val numberOfDocuments = ratings.count()

    //    ratings.show()

    val documentWords = ratings
      .select($"id", split($"document", "\\s+").as("words"))

    println("Enter a query")
    val prompt = scala.io.StdIn.readLine()

    val query = sparkSession
      .sparkContext
      .parallelize(prompt.split(" "))
      .map(s => (s, prompt.length))
      .toDF("token", "querySize")

    //    query.show()

    val explodedDocuments = documentWords
      .select($"id", explode($"words").as("token"), size($"words").as("documentSize"))

    val tf = query
      .join(explodedDocuments, Seq("token"), "left")
      .groupBy("id", "token", "documentSize")
      .agg(count("*").as("tf"))
      .select($"id", $"token", ($"tf" / $"documentSize").as("tf")) // Normalizes tf
      .cache()

    //    tf.show()

    val df = tf
      .groupBy("token")
      .agg(countDistinct("id").as("df"))

    //    df.show()

    val calcIdfUdf = udf { df: Long => math.log((numberOfDocuments + 1) / (df + 1)) }
    val idf = df
      .withColumn("idf", calcIdfUdf(col("df")))
      .cache()

    //    idf.show()

    val ratingsTfIdf = tf
      .join(idf, Seq("token"), "left")
      .select($"id", $"token", ($"tf" * $"idf").as("ratingsTfIdf"))

    //    ratingsTfIdf.show()

    val queryTf = query
      .groupBy($"token", $"querySize")
      .agg(count("*").as("tf"))
      .select($"token", ($"tf" / $"querySize").as("tf"))

    //    queryTf.show()

    val queryTfIdf = queryTf
      .join(idf, Seq("token"), "left")
      .select($"token", ($"tf" * $"idf").as("queryTfIdf"))
      .cache()

    //    queryTfIdf.show()

    val compareVectors = queryTfIdf
      .join(ratingsTfIdf, Seq("token"), "left")
      .cache()

    //    compareVectors.show()

    val ratingsMagnitude = compareVectors
      .groupBy($"id")
      .agg(sum(pow($"ratingsTfIdf", 2)).as("ratingsMagnitude"))
      .select($"id", sqrt($"ratingsMagnitude").as("ratingsMagnitude"))

    //    ratingsMagnitude.show()

    val queryMagnitude = math.sqrt(queryTfIdf
      .agg(sum(pow($"queryTfIdf", 2))).take(1)(0).getDouble(0))

    println("Query Magnitude", queryMagnitude)

    val dotProducts = compareVectors
      .groupBy($"id")
      .agg(sum($"queryTfIdf" * $"ratingsTfIdf").as("dot"))

    //    dotProducts.show()

    val calcCosine = udf { (dot: Double, ratingsMagnitude:Double) => dot / (ratingsMagnitude * queryMagnitude) }
    val cosineSimilarity = dotProducts
      .join(ratingsMagnitude, Seq("id"), "left")
      .withColumn("cosineSimilarity", calcCosine($"dot", $"ratingsMagnitude"))
      .select($"id", $"cosineSimilarity")

    //    cosineSimilarity.show()

    val result = cosineSimilarity
      .sort($"cosineSimilarity".desc)
      .limit(10)
      .join(ratings, Seq("id"), "left")

    result.show(false)

    scala.io.StdIn.readLine()

  }
}