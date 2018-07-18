package com.ankit.BigMartSales

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.ml.tuning.CrossValidator
import scala.collection.mutable.ListBuffer
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.regression.LinearRegression



object BMSLinearRegression {
  
  
  var regParams = Array(0,0.01,0.02,0.04,0.08,0.16,0.32,0.64,1.28,2.56,5.12,10.24)
  var maxIterations = Array(100,1000)
  
  var loss : ListBuffer[Double] = new ListBuffer()
  var cvLoss : ListBuffer[Double] = new ListBuffer()
  def bestModel(trainSet : DataFrame) : CrossValidatorModel={
    
   
    
    val lr = new LinearRegression().setFeaturesCol("features")
                         .setLabelCol("Item_Outlet_Sales")
                         .setPredictionCol("predictedLabel")
                         
    val evaluator = new RegressionEvaluator()
                          .setLabelCol("Item_Outlet_Sales")
                          .setPredictionCol("predictedLabel")
                          .setMetricName("rmse")
    
    val paramMaps = new ParamGridBuilder()
                           .addGrid(lr.regParam, regParams)
                           .addGrid(lr.maxIter, maxIterations)
                          .build()
    val cv = new CrossValidator().setEstimator(lr)
                  .setEvaluator(evaluator)
                  .setEstimatorParamMaps(paramMaps)
    val model = cv.fit(trainSet)
    return model;
    
  }
  
  
  
  
  
  def main(args : Array[String]) {
   val spark = SparkSession.builder().getOrCreate();
   import spark.implicits._
   val trainFile = spark.read.format("csv").option("header", "true").load(args(0));
   val testFile = spark.read.format("csv").option("header", "true").load(args(1))
   val dataPrep = new DataPreprator()
   var trainData = dataPrep.dataPreprator(trainFile)
   trainData = dataPrep.valuePreprator(trainData)
   trainData.persist();
   val testData = dataPrep.dataPreprator(testFile)
   
   val model =  bestModel(trainData);
   val rslt = model.transform(testData);
   val Array(trnData, cvData)= trainData.randomSplit(Array(0.7,0.3));
   val evaluator = new RegressionEvaluator()
                        .setLabelCol("Item_Outlet_Sales")
                        .setPredictionCol("predictedLabel")
                        .setMetricName("rmse")
    
    val trnAccuracy = evaluator.evaluate(model.transform(trnData))
    val cvAccuracy = evaluator.evaluate(model.transform(cvData))
    
                  
   
   
     
     rslt.select($"Item_Identifier", $"Outlet_Identifier", $"predictedLabel".as("Item_Outlet_Sales")).write.option("header",true).csv(args(2))                                
                                      
     
     println("Training Data Accuracy : "+trnAccuracy)
     println("CV Data Accuracy : "+cvAccuracy)
   
   
  }
  
  
}