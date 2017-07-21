import breeze.linalg.{linspace, DenseMatrix => BDM, DenseVector => BDV}
import breeze.plot._
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}

 /**
  * Created by THINKPAD on 2017/6/27.
  */
object script {

  def main(args: Array[String]) {

  val spark = SparkSession
    .builder()
      .master("local[8]")
    .appName("script")
    .config("spark.jars", "C:\\Codes\\IdeaProjects\\MachineLearning\\outJar\\MachineLearning.jar")
    .getOrCreate()
//    spark.driver.allowMultipleContexts = true

    //    一些config
    //    .enableHiveSupport()
    //    .config("spark.some.config.option", "some-value")
    //    spark.conf.set("spark.sql.shuffle.partitions", 6)
//    spark.conf.set("spark.executor.memory", "2g")
//    spark.conf.set()
//val sqlContext = new SQLContext(spark)
//读取csv
    val crimeDataPath = "C:\\Codes\\PyProject\\modelXYZ\\04-犯罪相关\\code\\BeijingCrime_2016.txt"
//    val crimeData = spark.read.format("org.apache.spark.sql.execution.datasources.csv.CSVFileFormat").option("header",  true).option("delimiter", ",").load(crimeDataPath)
//    crimeData.printSchema()
    //读取text
    val  data1 = spark.read.textFile(crimeDataPath).rdd
    println(data1.first())
    println(data1.count())
    val numcol = data1.map(f=>f.split(",").length).first()
    println(numcol)

//    try spark context
//    error:
/*    val conf = new SparkConf().setAppName("crimeModel")
      .setMaster("local[8]")
      .setJars(List("C:\\Codes\\IdeaProjects\\MachineLearning\\outJar\\MachineLearning.jar"))
    val sc = new SparkContext(conf)
    val data2 = sc.textFile(crimeDataPath)
    println("------------spark context---------------")
    println(data2.first())*/

    /*  try normalization
     val dataFrame = sqlContext.createDataFrame(Seq(
        (0, Vectors.dense(1.0, 0.5, -1.0)),
        (1, Vectors.dense(2.0, 1.0, 1.0)),
        (2, Vectors.dense(4.0, 10.0, 2.0))
      )).toDF("id", "features")
      // Normalize each Vector using $L^1$ norm.
      val normalizer = new Normalizer()
        .setInputCol("features")
        .setOutputCol("normFeatures")
        .setP(1.0)
      val l1NormData = normalizer.transform(dataFrame)
      println("Normalized using L^1 norm")
      l1NormData.show()

      // Normalize each Vector using $L^\infty$ norm.
      val lInfNormData = normalizer.transform(dataFrame, normalizer.p -> Double.PositiveInfinity)
      println("Normalized using L^inf norm")
      lInfNormData.show()*/
    /*
        //try estimator, transformer, param

        // Prepare training data from a list of (label, features) tuples.
        val training = sqlContext.createDataFrame(Seq(
          (1.0, Vectors.dense(0.0, 1.1, 0.1)),
          (0.0, Vectors.dense(2.0, 1.0, -1.0)),
          (0.0, Vectors.dense(2.0, 1.3, 1.0)),
          (1.0, Vectors.dense(0.0, 1.2, -0.5))
        )).toDF("label", "features")

        // Create a LogisticRegression instance.  This instance is an Estimator.
        val lr = new LogisticRegression()
        // Print out the parameters, documentation, and any default values.
    //    println("LogisticRegression parameters:\n" + lr.explainParams() + "\n")

    /*    printed:
    LogisticRegression parameters:
          elasticNetParam: the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty (default: 0.0)
        featuresCol: features column name (default: features)
        fitIntercept: whether to fit an intercept term (default: true)
        labelCol: label column name (default: label)
        maxIter: maximum number of iterations (>= 0) (default: 100)
        predictionCol: prediction column name (default: prediction)
        probabilityCol: Column name for predicted class conditional probabilities. Note: Not all models output well-calibrated probability estimates! These probabilities should be treated as confidences, not precise probabilities (default: probability)
        rawPredictionCol: raw prediction (a.k.a. confidence) column name (default: rawPrediction)
        regParam: regularization parameter (>= 0) (default: 0.0)
        standardization: whether to standardize the training features before fitting the model (default: true)
        threshold: threshold in binary classification prediction, in range [0, 1] (default: 0.5)
        thresholds: Thresholds in multi-class classification to adjust the probability of predicting each class. Array must have length equal to the number of classes, with values >= 0. The class with largest value p/t is predicted, where p is the original probability of that class and t is the class' threshold. (undefined)
        tol: the convergence tolerance for iterative algorithms (default: 1.0E-6)
        weightCol: weight column name. If this is not set or empty, we treat all instance weights as 1.0. (default: )*/
        // We may set parameters using setter methods.
        lr.setMaxIter(10)
          .setRegParam(0.01)

        // Learn a LogisticRegression model.  This uses the parameters stored in lr.
        val model1 = lr.fit(training)
        // Since model1 is a Model (i.e., a Transformer produced by an Estimator),
        // we can view the parameters it used during fit().
        // This prints the parameter (name: value) pairs, where names are unique IDs for this
        // LogisticRegression instance.
        println("Model 1 was fit using parameters: " + model1.parent.extractParamMap)

        // We may alternatively specify parameters using a ParamMap,
        // which supports several methods for specifying parameters.
        val paramMap = ParamMap(lr.maxIter -> 20)
          .put(lr.maxIter, 30) // Specify 1 Param.  This overwrites the original maxIter.
          .put(lr.regParam -> 0.1, lr.threshold -> 0.55) // Specify multiple Params.

        // One can also combine ParamMaps.
        val paramMap2 = ParamMap(lr.probabilityCol -> "myProbability") // Change output column name
        val paramMapCombined = paramMap ++ paramMap2

        // Now learn a new model using the paramMapCombined parameters.
        // paramMapCombined overrides all parameters set earlier via lr.set* methods.
        val model2 = lr.fit(training, paramMapCombined)
        println("Model 2 was fit using parameters: " + model2.parent.extractParamMap)

        // Prepare test data.
        val test = sqlContext.createDataFrame(Seq(
          (1.0, Vectors.dense(-1.0, 1.5, 1.3)),
          (0.0, Vectors.dense(3.0, 2.0, -0.1)),
          (1.0, Vectors.dense(0.0, 2.2, -1.5))
        )).toDF("label", "features")

        // Make predictions on test data using the Transformer.transform() method.
        // LogisticRegression.transform will only use the 'features' column.
        // Note that model2.transform() outputs a 'myProbability' column instead of the usual
        // 'probability' column since we renamed the lr.probabilityCol parameter previously.
        model2.transform(test)
          .select("features", "label", "myProbability", "prediction")
          .collect()
          .foreach { case Row(features: Vector, label: Double, prob: Vector, prediction: Double) =>
            println(s"($features, $label) -> prob=$prob, prediction=$prediction")
          }*/



/*
    //尝试画图

    //    val a = new BDV[Int](1 to 3 toArray)
    //    val b = new BDM[Int](3, 3, 1 to 9 toArray)
    val f = Figure()
    val p = f.subplot(0)
    val x = linspace(0.0,1.0)
    p += plot(x, x :^ 2.0)
    p += plot(x, x :^ 3.0, '.')
    p.xlabel = "x axis"
    p.ylabel = "y axis"
    //    f.saveas("lines.png")
    val p2 = f.subplot(2,1,1)
    val g = breeze.stats.distributions.Gaussian(0,1)
    p2 += hist(g.sample(100000),100)
    p2.title = "A normal distribution"
//    f.saveas("subplots.png")*/


    spark.stop()
  }

}
