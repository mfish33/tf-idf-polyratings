import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, count, countDistinct, explode, pow, size, split, sqrt, sum, udf}

object tfidf {
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

    val tf = documentWords
      .select($"id", explode($"words").as("token"), size($"words").as("documentSize"))
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
      .cache()

    ratingsTfIdf.count() // Force spark to evaluate

//    ratingsTfIdf.show()

    println("Enter a query")
    val prompt = scala.io.StdIn.readLine()

    val queryWords = prompt.split(" ")
    val query = sparkSession
      .sparkContext
      .parallelize(queryWords)
      .map(s => (s, queryWords.length))
      .toDF("token", "querySize")

//    query.show()

    val queryTf = query
      .groupBy($"token", $"querySize")
      .agg(count("*").as("tf"))
      .select($"token", ($"tf" / $"querySize").as("tf"))

//    queryTf.show()

    val queryTfIdf = queryTf
      .join(idf, Seq("token"), "inner")
      .select($"token", ($"tf" * $"idf").as("queryTfIdf"))

//    queryTfIdf.show()

    val compareVectors = queryTfIdf
      .join(ratingsTfIdf, Seq("token"), "left")
      .cache()

//    compareVectors.show()

    val ratingsMagnitude = compareVectors
      .groupBy($"id")
      .agg(sum(pow($"ratingsTfIdf", 2)).as("ratingsMagnitude"))
      .filter($"ratingsMagnitude" > 0) // Don't care if the magnitude is zero since there is no similarity
      .select($"id", sqrt($"ratingsMagnitude").as("ratingsMagnitude"))

//    ratingsMagnitude.show()

    val dotProducts = compareVectors
      .groupBy($"id")
      .agg(sum($"queryTfIdf" * $"ratingsTfIdf").as("dot"))
      .filter($"dot" > 0) // Remove all that are zero since no similarity

//    dotProducts.show()

    val calcCosine = udf { (dot: Double, ratingsMagnitude:Double) => dot / ratingsMagnitude }
    val cosineSimilarity = dotProducts
      .join(ratingsMagnitude, Seq("id"), "inner")
      .withColumn("cosineSimilarity", calcCosine($"dot", $"ratingsMagnitude"))
      .select($"id", $"cosineSimilarity")

//    cosineSimilarity.show()

    val result = cosineSimilarity
      .sort($"cosineSimilarity".desc)
      .limit(10)
      .join(ratings, Seq("id"), "left")
      .sort($"cosineSimilarity".desc)

    result.show(false)
  }
}