import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.linalg.Vectors

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.sql.types.DataTypes

object TTTMRandomForest {
  def main(args: Array[String]) {
    val jobID = args(0)
    val dt = args(0).substring(0, 8)

    val spark = SparkSession
      .builder
      .appName("TTTM")
      .config("spark.sql.warehouse.dir", "/fcbig/warehouse")
      .config("hive.metastore.uris", "thrift://fcbig-06-12:9083")
      .enableHiveSupport()
      .getOrCreate()

    val TTTMFile = spark.read.option("header", "true")
      .option("inferSchema", true)
      .option("sep", "\t")
      .csv("/fcbig/" + jobID)

    TTTMFile.show(false)

    val colNameList = TTTMFile.columns.toList

    TTTMFile.columns.toList.map( x => TTTMFile.withColumn(x, TTTMFile.col(x).cast(DataTypes.DoubleType)))

    TTTMFile.withColumn("Label", TTTMFile.col("Label").cast(DataTypes.DoubleType))

    val assembler = new VectorAssembler()
      .setInputCols(TTTMFile.columns.drop(1))
      .setOutputCol("features")
      
    val output = assembler.transform(TTTMFile)
    
    val labelIndexer = new StringIndexer()
      .setInputCol("Label")
      .setOutputCol("indexedLabel")
      .fit(output)
    

    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(output)

    val rf = new RandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setMaxDepth(30)
      .setNumTrees(500)

    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

    val model = pipeline.fit(output)

    val result = model
      .stages(2)
      .asInstanceOf[RandomForestClassificationModel]
      .featureImportances
      .toArray

    var insertSQLString = "INSERT INTO bizanal.tttm PARTITION(dt='" + dt + "') VALUES "
    var valueString = ""
    for (i <- 0 until result.length) {
        println(colNameList(i + 1) + "        " + result(i))
        valueString += "('" + jobID + "', '" + colNameList(i + 1) + "', " + result(i).toString() + "),"
    }

    insertSQLString = insertSQLString + valueString.dropRight(1)

    spark.sqlContext.sql(insertSQLString)

  }
}