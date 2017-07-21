/**
  * Created by GAO SHUO on 2017/7/4.
  * try cluster subway cards of Shanghai
  */
import org.apache.spark.ml.linalg.DenseVector
//import org.apache.spark.ml.linalg.{Vector, Vectors}
//import org.apache.spark.ml.clustering.GaussianMixture
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession}
//import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.ml.clustering.KMeans
import breeze.linalg.{Vector=>BV}

object subway {
  def main(args: Array[String]) {
    // spark session
    val spark = SparkSession
      .builder()
      .master("local[8]")
      .appName("script")
      .config("spark.jars", "C:\\Codes\\IdeaProjects\\MachineLearning\\outJar\\MachineLearning.jar")
      .getOrCreate()
    import spark.implicits._

    //    spark context
    //    spark.sparkContext  .textFile("examples/src/main/resources/people.txt")

    //    read card data
    val dataPath = "D:\\data\\temp\\subway\\ShanghaiSubway\\user_7DaysHours_count_pivot"
    var df = spark.read.format("org.apache.spark.sql.execution.datasources.csv.CSVFileFormat").option("header",  true).option("delimiter", ",").load(dataPath)
    //    df.show(10)
    //    df.printSchema()
    // convert to double
    val col = df.columns
    for(i<-1 until col.length){
      df =  df.withColumn(col(i), df(col(i)).cast(DoubleType))
    }
    //    replace null with 0
    df = df.na.fill(0.0)

    //    convert features to vector
    // Turn it into an RDD for manipulation.
    val inputRDD: RDD[(String, DenseVector)] =
      df.map(row =>{
        val num = row.length
        val  bb = ArrayBuffer[Double]()
        for (i<- 1 until num){
          bb+= row.getAs[Double](i )
        }
        (row.getAs[String]("cardId") ,new DenseVector  (bb.toArray ))
      }).rdd

    val inputDF = inputRDD.toDF("ID","features" )

    //  take sample from data frame
    // Register the DataFrame as a temporary view
    inputDF.createOrReplaceTempView("data")
    val sampleDF = spark.sql("SELECT   * FROM data limit 1000")
    //    sampleDF.show(10)
    //    sampleDF.printSchema()

    // Trains Gaussian Mixture Model
    /*    val gmm = new GaussianMixture()
          .setK(3)
        val model = gmm.fit(sampleDF)
        val outputModel = model.transform(sampleDF)
        println("---------------")
         // output parameters of mixture model model
        for (i <- 0 until model.getK) {
          println(s"Gaussian $i:\nweight=${model.weights(i)}\n" +
            s"mu=${model.gaussians(i).mean}\nsigma=\n${model.gaussians(i).cov}\n")
        }
    //    calculate the BIC
        outputModel.createOrReplaceTempView("data1")
        val sampleoutput = spark.sql("SELECT   * FROM data1 where prediction >0")
        sampleoutput.show(100)*/


    // Trains a k-means model.
    // decide k
    /*  val arrayOfRows = new ArrayBuffer[String]
      for (i<- 2 until 10){
        val kmeans = new KMeans().setK(i).setSeed(1L)
        val model = kmeans.fit(inputDF)
        val WSSSE = model.computeCost(inputDF)
        arrayOfRows += i+"\t"+WSSSE
      }
      println(arrayOfRows.mkString("----"))*/
    //    result:
    //    sample 1000
    // 7	5337.6451495963465----8	5335.866528783665
    //  all
    //  2	2.378168070725338E7----3	2.2193864188881494E7----4	2.1365600967408136E7----5	2.091523600484477E7----6	2.076042051923335E7----7	2.0818338677193135E7----8	2.067267996861386E7----9	2.014502062506387E7
    //   set k=6
    val kmeans = new KMeans().setK(6).setSeed(1L)
    val model = kmeans.fit(inputDF)

    // Shows the result.
    println("Cluster Centers: ")
    model.clusterCenters.foreach(println)
    val outputModel = model.transform(inputDF)
    outputModel.show(10)
        val dest = "D:\\data\\temp\\subway\\ShanghaiSubway\\kmeanscluster"
    val outputRdd = outputModel.rdd.map(f=>(f.getString(0) , f.getAs[DenseVector](1) , f.getInt(2))).
//      用户ID，feature向量，clusterID
      map(f=>(f._3 , f._2.toArray))
// 每个cluster的样本个数
    val op1 = outputRdd.map(f=>(f._1,1)).reduceByKey(_+_)
      op1.map(f=>f._1 +"\t"+f._2).repartition(1).saveAsTextFile(dest+"_usercount")
//每个cluster中feature行程次数总和
      val op2 = outputRdd.reduceByKey((a,b)=>{
      val n = a.length
      val c = new Array[Double](n)
      for (i<- 0 until n){
        c(i) = a(i) +b(i)
      }
      c
    })
    //每个cluster中人均feature行程次数
    val op3 = op2.join(op1).map(f=>(f._1, f._2._1.map(s=>s/f._2._2)))
//      .groupByKey(50).map(f=>(f._1,f._2.toArray.length,summaryProb(f._2..sum.  .toArray)))

//    output cluster and user count
//    outputRdd.map(f=>f._1 +"\t"+f._2).saveAsTextFile(dest+"_usercount")
//    val clusters = outputRdd.map(f=>f._1).collect()
    val clusters =Array(0,1,2,3,4,5)
    /* |-- cardId: string (nullable = true)
 |-- 星期一_04: double (nullable = false)
 |-- 星期一_05: double (nullable = false)
 |-- 星期一_06: double (nullable = false)
 |-- 星期一_07: double (nullable = false)
 |-- 星期一_08: double (nullable = false)
 |-- 星期一_09: double (nullable = false)
 |-- 星期一_10: double (nullable = false)
 |-- 星期一_11: double (nullable = false)
 |-- 星期一_12: double (nullable = false)
 |-- 星期一_13: double (nullable = false)
 |-- 星期一_14: double (nullable = false)
 |-- 星期一_15: double (nullable = false)
 |-- 星期一_16: double (nullable = false)
 |-- 星期一_17: double (nullable = false)
 |-- 星期一_18: double (nullable = false)
 |-- 星期一_19: double (nullable = false)
 |-- 星期一_20: double (nullable = false)
 |-- 星期一_21: double (nullable = false)
 |-- 星期一_22: double (nullable = false)
 |-- 星期一_23: double (nullable = false)
 |-- 星期三_00: double (nullable = false)
 |-- 星期三_04: double (nullable = false)
 |-- 星期三_05: double (nullable = false)
 |-- 星期三_06: double (nullable = false)
 |-- 星期三_07: double (nullable = false)
 |-- 星期三_08: double (nullable = false)
 |-- 星期三_09: double (nullable = false)
 |-- 星期三_10: double (nullable = false)
 |-- 星期三_11: double (nullable = false)
 |-- 星期三_12: double (nullable = false)
 |-- 星期三_13: double (nullable = false)
 |-- 星期三_14: double (nullable = false)
 |-- 星期三_15: double (nullable = false)
 |-- 星期三_16: double (nullable = false)
 |-- 星期三_17: double (nullable = false)
 |-- 星期三_18: double (nullable = false)
 |-- 星期三_19: double (nullable = false)
 |-- 星期三_20: double (nullable = false)
 |-- 星期三_21: double (nullable = false)
 |-- 星期三_22: double (nullable = false)
 |-- 星期三_23: double (nullable = false)
 |-- 星期二_04: double (nullable = false)
 |-- 星期二_05: double (nullable = false)
 |-- 星期二_06: double (nullable = false)
 |-- 星期二_07: double (nullable = false)
 |-- 星期二_08: double (nullable = false)
 |-- 星期二_09: double (nullable = false)
 |-- 星期二_10: double (nullable = false)
 |-- 星期二_11: double (nullable = false)
 |-- 星期二_12: double (nullable = false)
 |-- 星期二_13: double (nullable = false)
 |-- 星期二_14: double (nullable = false)
 |-- 星期二_15: double (nullable = false)
 |-- 星期二_16: double (nullable = false)
 |-- 星期二_17: double (nullable = false)
 |-- 星期二_18: double (nullable = false)
 |-- 星期二_19: double (nullable = false)
 |-- 星期二_20: double (nullable = false)
 |-- 星期二_21: double (nullable = false)
 |-- 星期二_22: double (nullable = false)
 |-- 星期二_23: double (nullable = false)
 |-- 星期五_04: double (nullable = false)
 |-- 星期五_05: double (nullable = false)
 |-- 星期五_06: double (nullable = false)
 |-- 星期五_07: double (nullable = false)
 |-- 星期五_08: double (nullable = false)
 |-- 星期五_09: double (nullable = false)
 |-- 星期五_10: double (nullable = false)
 |-- 星期五_11: double (nullable = false)
 |-- 星期五_12: double (nullable = false)
 |-- 星期五_13: double (nullable = false)
 |-- 星期五_14: double (nullable = false)
 |-- 星期五_15: double (nullable = false)
 |-- 星期五_16: double (nullable = false)
 |-- 星期五_17: double (nullable = false)
 |-- 星期五_18: double (nullable = false)
 |-- 星期五_19: double (nullable = false)
 |-- 星期五_20: double (nullable = false)
 |-- 星期五_21: double (nullable = false)
 |-- 星期五_22: double (nullable = false)
 |-- 星期五_23: double (nullable = false)
 |-- 星期六_00: double (nullable = false)
 |-- 星期六_04: double (nullable = false)
 |-- 星期六_05: double (nullable = false)
 |-- 星期六_06: double (nullable = false)
 |-- 星期六_07: double (nullable = false)
 |-- 星期六_08: double (nullable = false)
 |-- 星期六_09: double (nullable = false)
 |-- 星期六_10: double (nullable = false)
 |-- 星期六_11: double (nullable = false)
 |-- 星期六_12: double (nullable = false)
 |-- 星期六_13: double (nullable = false)
 |-- 星期六_14: double (nullable = false)
 |-- 星期六_15: double (nullable = false)
 |-- 星期六_16: double (nullable = false)
 |-- 星期六_17: double (nullable = false)
 |-- 星期六_18: double (nullable = false)
 |-- 星期六_19: double (nullable = false)
 |-- 星期六_20: double (nullable = false)
 |-- 星期六_21: double (nullable = false)
 |-- 星期六_22: double (nullable = false)
 |-- 星期六_23: double (nullable = false)
 |-- 星期四_00: double (nullable = false)
 |-- 星期四_04: double (nullable = false)
 |-- 星期四_05: double (nullable = false)
 |-- 星期四_06: double (nullable = false)
 |-- 星期四_07: double (nullable = false)
 |-- 星期四_08: double (nullable = false)
 |-- 星期四_09: double (nullable = false)
 |-- 星期四_10: double (nullable = false)
 |-- 星期四_11: double (nullable = false)
 |-- 星期四_12: double (nullable = false)
 |-- 星期四_13: double (nullable = false)
 |-- 星期四_14: double (nullable = false)
 |-- 星期四_15: double (nullable = false)
 |-- 星期四_16: double (nullable = false)
 |-- 星期四_17: double (nullable = false)
 |-- 星期四_18: double (nullable = false)
 |-- 星期四_19: double (nullable = false)
 |-- 星期四_20: double (nullable = false)
 |-- 星期四_21: double (nullable = false)
 |-- 星期四_22: double (nullable = false)
 |-- 星期四_23: double (nullable = false)
 |-- 星期日_00: double (nullable = false)
 |-- 星期日_02: double (nullable = false)
 |-- 星期日_04: double (nullable = false)
 |-- 星期日_05: double (nullable = false)
 |-- 星期日_06: double (nullable = false)
 |-- 星期日_07: double (nullable = false)
 |-- 星期日_08: double (nullable = false)
 |-- 星期日_09: double (nullable = false)
 |-- 星期日_10: double (nullable = false)
 |-- 星期日_11: double (nullable = false)
 |-- 星期日_12: double (nullable = false)
 |-- 星期日_13: double (nullable = false)
 |-- 星期日_14: double (nullable = false)
 |-- 星期日_15: double (nullable = false)
 |-- 星期日_16: double (nullable = false)
 |-- 星期日_17: double (nullable = false)
 |-- 星期日_18: double (nullable = false)
 |-- 星期日_19: double (nullable = false)
 |-- 星期日_20: double (nullable = false)
 |-- 星期日_21: double (nullable = false)
 |-- 星期日_22: double (nullable = false)
 |-- 星期日_23: double (nullable = false) */
//    output user probability for each cluster
    for(i<- clusters){
      val arr = op3.filter(f=>f._1 == i & f._2.length ==145).map(f=>f._2).first()
      val timeDays1 = Array(0.0,0.0,0.0,0.0) ++ arr.take(20)
      val timeDays2 = Array(0.0,0.0,0.0,0.0) ++ arr.take(61).takeRight(20)
      val timeDays3 = Array(arr(20)) ++ Array(0.0,0.0,0.0) ++ arr.take(41).takeRight(20)
      val timeDays4 = Array(arr(102)) ++ Array(0.0,0.0,0.0) ++ arr.take(123).takeRight(20)
      val timeDays5 = Array(0.0,0.0,0.0,0.0)  ++ arr.take(81).takeRight(20)
      val timeDays6 = Array(arr(81)) ++Array(0.0,0.0,0.0 )  ++ arr.take(102).takeRight(20)
      val timeDays7 = Array(arr(123)) ++Array(0.0)++  Array(arr(124)) ++Array(0.0) ++ arr.takeRight(20)
      val allDays = Array(timeDays1,timeDays2 ,timeDays3 ,timeDays4 ,timeDays5 ,timeDays6 ,timeDays7 )
      spark.sparkContext.parallelize(allDays).map(_.mkString("\t")).repartition(1).saveAsTextFile(dest+"_"+i)
    }

  }
//  unused
    def summaryProb(a:Array[Array[Double]]) = {
    val nRow = a.length
    val nCol = a(0).length
    val op = new Array[Double](nCol)
    for(i<- 0 until nCol){
      val b =  new Array[Double](nRow)
      for (j<- 0 until nRow){
        b(j) = a(j)(i)
      }
      op(i) = b.sum/b.length
    }
    for (i<-op.indices){
      op(i) = op(i)/op.sum
    }
    op
  }



}
