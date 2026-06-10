import breeze.linalg.DenseVector
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.feature.{PCA, StandardScaler, VectorAssembler}
import org.apache.spark.ml.feature.StandardScaler

case class OutputRow(_c0: Int, pcaFeatures: breeze.linalg.DenseVector[Double])

object TTTMPCA {
  def main(args: Array[String]) {
    val spark = SparkSession
      .builder
      .appName("TTTM")
      .config("spark.sql.warehouse.dir", "/fcbig/warehouse")
      .config("hive.metastore.uris", "thrift:/fcbig-06-12:9083")
      .enableHiveSupport()
      .getOrCreate()

    val jobID = args(0)

    val PCAFile = spark.read
      .option("inferSchema", true)
      .csv("/fcbig/pca/" + jobID)
      .na.drop("any")

    val colNames = PCAFile.columns.toBuffer - "_c0"

    val assembler = new VectorAssembler()
      .setInputCols(colNames.toArray)
      .setOutputCol("features")

    val output = assembler.transform(PCAFile)

    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")

    val scalerDF = scaler.fit(output).transform(output)

    val pca = new PCA()
      .setInputCol("scaledFeatures")
      .setOutputCol("pcaFeatures")
      .setK(2)
      .fit(scalerDF)

    import spark.implicits._

    val result = pca.transform(scalerDF).select("_c0", "pcaFeatures")
      .map (
        row => {
          val id = row.getAs[Int](0)
          val features = row.getAs[org.apache.spark.ml.linalg.DenseVector](1).toArray
          (id, features(0).toString(), features(1).toString)
        }
      )          
      .repartition(1)
      .write
      .format("csv")
      .save("/fcbig/output/" + jobID)

    }
}