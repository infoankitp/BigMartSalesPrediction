package com.ankit.BigMartSales

import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.types.IntegerType

import org.apache.spark.sql
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.StandardScaler

class DataPreprator {
  
  def dataPreprator(df : DataFrame) : DataFrame = {
    val spark = SparkSession.builder().getOrCreate();
    import spark.implicits._;    
    var rslt = df.withColumn("Item_MRP", $"Item_MRP".cast(DoubleType)).na.fill(0)
    
    rslt = rslt.withColumn("Item_Weight", $"Item_Weight".cast(DoubleType)).na.fill(0)
    rslt = rslt.withColumn("Item_Visibility", $"Item_Visibility".cast(DoubleType)).na.fill(0)
    rslt = rslt.withColumn("Outlet_Establishment_Year", $"Outlet_Establishment_Year".cast(IntegerType)).na.fill(0)
    
    rslt = rslt.withColumn("Item_Category1",substring($"Item_Identifier",0,2))
    rslt = rslt.withColumn("Item_Fat_Content",when($"Item_Fat_Content"==="low fat" or $"Item_Fat_Content"==="Low Fat" or $"Item_Fat_Content"==="LF","Low")
                                                                                                    .when($"Item_Fat_Content"==="Regular" or $"Item_Fat_Content"==="reg","Regular"))
                                                                                                    
    val avgVisibility = rslt.agg(avg("Item_Visibility")).head.getDouble(0)
    rslt = rslt.withColumn("Item_Visibility",when($"Item_Visibility"===0,avgVisibility).otherwise($"Item_Visibility"))       
    
    
    val itemFatContentIndexer = new StringIndexer()
                .setInputCol("Item_Fat_Content")
                .setOutputCol("Item_Fat_Content_Index")
                      .fit(rslt)
    rslt = itemFatContentIndexer.transform(rslt)
    
    rslt = rslt.withColumn("Item_Category1",substring($"Item_Identifier",0,2))
    
    val Item_Category1Indexer = new StringIndexer()
                .setInputCol("Item_Category1")
                .setOutputCol("Item_Category1_Index")
                      .fit(rslt)
    rslt = Item_Category1Indexer.transform(rslt)
    
    val Item_TypeIndexer = new StringIndexer()
                .setInputCol("Item_Type")
                .setOutputCol("Item_Type_Index")
                      .fit(rslt)
    rslt = Item_TypeIndexer.transform(rslt)
    
    rslt = rslt.withColumn("Age",($"Outlet_Establishment_Year"-2013)*(-1))
    
    val Outlet_Location_TypeIndexer = new StringIndexer()
                .setInputCol("Outlet_Location_Type")
                .setOutputCol("Outlet_Location_Type_Index")
                      .fit(rslt)
    rslt = Outlet_Location_TypeIndexer.transform(rslt)
    
    val Outlet_TypeIndexer =  new StringIndexer()
                .setInputCol("Outlet_Type")
                .setOutputCol("Outlet_Type_Index")
                      .fit(rslt)
    rslt = Outlet_TypeIndexer.transform(rslt)
    
    val featureCols = Array("Item_MRP","Item_Visibility","Age","Outlet_Location_Type_Index")
    val integrator = new VectorAssembler()
                              .setInputCols(featureCols)
                              .setOutputCol("features")
    rslt = integrator.transform(rslt)
    
    return rslt;
        
}


def valuePreprator(df : DataFrame) : DataFrame = {
    val rslt = df.withColumn("Item_Outlet_Sales", col("Item_Outlet_Sales").cast(DoubleType)).na.fill(0)
    return rslt; 
    
}
}