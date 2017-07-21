import org.apache.spark.{SparkConf, SparkContext, ml}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.feature.Normalizer
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame,  SQLContext}
import com.databricks.spark.csv._
import org.apache.spark.sql.SparkSession

import breeze.linalg.{ linspace, DenseMatrix => BDM, DenseVector => BDV}
import breeze.numerics._
import scala.collection.mutable.ArrayBuffer
import scala.io._
import breeze.plot._
//import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
//import org.apache.spark.implicits._
import org.apache.commons.csv.CSVFormat
/**
  * Created by Gao Shuo on 2017/6/28.
  */
object crimeModel {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("crimeModel")
      .setMaster("local[8]")
      .setJars(List("C:\\Codes\\IdeaProjects\\MachineLearning\\outJar\\MachineLearning.jar"))
    val sc = new SparkContext(conf)
    val sqlContext=new SQLContext(sc)
    val crimeDataPath = "C:\\Codes\\PyProject\\modelXYZ\\04-犯罪相关\\code\\BeijingCrime_2016.txt"
    val crimeData=sqlContext.csvFile(filePath=crimeDataPath, useHeader=true, delimiter=',')
    crimeData.printSchema()
    //    crimeData.show(numRows = 30)
    //    crimeData.take(5).foreach(println)
    //    val areaDataFrame: DataFrame = crimeData.select("KmArea")
    //    areaDataFrame.show()
    //    crimeData.filter("KmArea >2").show(3)
    //    crimeData.sort(crimeData("KmArea").desc).show(5)

    val columnName= crimeData.columns.zipWithIndex.map(f=>f._2+","+f._1)
    println(columnName.mkString("\t"))

//    新建一个Array，保存features
    val features = new ArrayBuffer[Int]()
//    val area = BDV( Source.fromFile(crimeDataPath)
//      .getLines.drop(1).map(_.split(",")(0).toDouble).toSeq :_ * )
//    尝试画图
//    val f = Figure()
//    features += crimeData.columns(0)
//
//     val HomeP = BDV( Source.fromFile(crimeDataPath)
//      .getLines.drop(1).map(f=>log(f.split(",")(2).toDouble)).toSeq :_ * )
//    f.clear()
//    val p1=  f.subplot(0)
//    p1 +=hist(HomeP,50)
//    p1.title = "HomeP distribution"
    features += 47
    for (i <- 0 to 46) features += i
   val enquiry = features.toArray.map(f=>crimeData.columns(f))
//     .mkString(",")
//    val featuresDF: DataFrame = crimeData.select(enquiry)
//    featuresDF.show()


  }

}
