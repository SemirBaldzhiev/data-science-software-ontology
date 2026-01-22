from owlready2 import *

onto = get_ontology("http://test.org/dsso.owl")

with onto:
    class Software(Thing): 
        pass
    
    class DataScienceSoftware(Software):
        pass
    
    class MLLibrary(DataScienceSoftware):
        pass
    
    class DataVisualizatioTool(DataScienceSoftware):
        pass
    
    class NotebookEnvironment(DataScienceSoftware):
        pass
    
    class CloudPlatform(DataScienceSoftware):
        pass
    
    class Algorithm(Thing):
        pass
    
    class MachineLearningMethod(Algorithm):
        pass
    
    class SupervisedLearningMethod(MachineLearningMethod):
        pass
    
    class UnsupervisedLearningMethod(MachineLearningMethod):
        pass
    
    class ReinforcementLearningMethod(MachineLearningMethod):
        pass
        
    class OptimizationAlgorithm(Algorithm):
        pass
    
    class HardwareResource(Thing):
        pass    
    
    class HardwareAccelerator(HardwareResource):
        pass
    
    class InstructionSet(HardwareResource):
        pass
    
    class License(Thing):
        pass
    
    class OpenSourceLicense(License):
        pass
    
    class ProprietaryLicense(License):
        pass
     
    class ModelArchitecture(Thing):
        pass
    
    class NeuralNetwork(ModelArchitecture):
        pass
    
    class FeedForward(NeuralNetwork):
        pass
    
    class Reccurent(NeuralNetwork):
        pass
    
    class Convolutional(NeuralNetwork):
        pass
    
    class Transformer(NeuralNetwork):
        pass
    
    class ProgrammingLanguage(Thing):
        pass
    
    class DataFormat(Thing):
        pass
    
    class Metric(Thing):
        pass
    
    class uses_architecture(ObjectProperty):
        domain = [Algorithm]
        range = [ModelArchitecture]
    
    class DeepLearningMethod(MachineLearningMethod):
        equivalent_to = [
            MachineLearningMethod & uses_architecture.some(NeuralNetwork)
        ]
    
    class Organization(Thing):
        pass
    
    class OperatingSystem(Thing):
        pass
    
    class has_license(ObjectProperty, FunctionalProperty):
        domain = [Software]
        range = [License]
        
    class depends_on(ObjectProperty, TransitiveProperty):
        domain = [Software]
        range = [Software]

    class requires_dependency(depends_on):
        pass
    
    class implements_algorithm(ObjectProperty):
        domain = [Software]
        range = [Algorithm]

    class is_implemented_by(ObjectProperty, InverseFunctionalProperty):
        domain = [Algorithm]
        range = [Software]
        inverse_property = implements_algorithm

    class developed_by(ObjectProperty):
        domain = [Software]
        range = [Organization]

    class runs_on_os(ObjectProperty):
        domain = [Software]
        range = [OperatingSystem]

    class written_in(ObjectProperty):
        domain = [Software]
        range = [ProgrammingLanguage]

    class supports_format(ObjectProperty):
        domain = [Software]
        range = [DataFormat]

    class accelerated_by(ObjectProperty):
        domain = [Software]
        range = [HardwareAccelerator]
    
    class optimizes_metric(ObjectProperty):
        domain = [Algorithm]
        range = [Metric]
    
    class interoperates_with(ObjectProperty, SymmetricProperty):
        domain = [Software]
        range = [Software]
    
    class has_version(DataProperty, FunctionalProperty):
        domain = [Software]
        range = [str]
    
    class is_deprecated(DataProperty, FunctionalProperty):
        domain = [Software]
        range = [bool]
    
    class released_in_year(DataProperty):
        domain = [Software]
        range = [int]

    class PurePythonTool(Software):
        pass

    class ApacheProject(Software):
        pass

    class DeepLearningSoftware(Software):
        equivalent_to = [
            Software & implements_algorithm.some(DeepLearningMethod)
        ]

    class EnterprisePlatform(Software):
        pass
    
    class OpenSourceSoftware(Software):
        equivalent_to = [
            Software & has_license.some(OpenSourceLicense)
        ]

    python = ProgrammingLanguage("Python")
    r_lang = ProgrammingLanguage("R")
    scala = ProgrammingLanguage("Scala")
    cpp = ProgrammingLanguage("Cpp")
    
    linux = OperatingSystem("Linux")
    windows = OperatingSystem("Windows")
    macOS = OperatingSystem("macOS")
    
    mit = OpenSourceLicense("MIT_License")
    apache2 = OpenSourceLicense("Apache_2.0")
    proprietary = ProprietaryLicense("Enterprise_License")
    
    google = Organization("Google")
    meta = Organization("Meta")
    apache = Organization("Apache_Foundation")
    microsoft = Organization("Microsoft")
    databricks_inc = Organization("Databricks_Inc")
    
    nvidia_A100 = HardwareAccelerator("NVIDIA_A100")
    google_TPU = HardwareAccelerator("Google_TPU_v4")
    
    csv = DataFormat("CSV")
    parquet = DataFormat("Parquet")
    onnx = DataFormat("ONNX")
    
    accuracy = Metric("Accuracy")
    f1_score = Metric("F1_Score")

    transformer_arch = Transformer("Transformer_Architecture")
    cnn_arch = Convolutional("CNN_Architecture")
    
    gpt_algo = MachineLearningMethod("GPT_Algorithm")
    gpt_algo.uses_architecture.append(transformer_arch)
    gpt_algo.optimizes_metric.append(accuracy)
    
    res_net_algo = MachineLearningMethod("ResNet_Algorithm")
    res_net_algo.uses_architecture.append(cnn_arch)
    
    random_forest = SupervisedLearningMethod("RandomForest")
    k_means = UnsupervisedLearningMethod("K_Means_Clustering")

    tf = MLLibrary("TensorFlow")
    tf.developed_by.append(google)
    tf.has_license = apache2
    tf.written_in.extend([python, cpp])
    tf.accelerated_by.extend([nvidia_A100, google_TPU])
    tf.implements_algorithm.append(res_net_algo)
    tf.supports_format.append(onnx)
    tf.released_in_year.append(2015)
    
    pt = MLLibrary("PyTorch")
    pt.developed_by.append(meta)
    pt.has_license = mit
    pt.written_in.extend([python, cpp])
    pt.accelerated_by.append(nvidia_A100)
    pt.implements_algorithm.append(gpt_algo)
    pt.interoperates_with.append(tf)
    
    sklearn = MLLibrary("Scikit_Learn")
    sklearn.has_license = mit
    sklearn.written_in.append(python)
    sklearn.implements_algorithm.extend([random_forest, k_means])
    sklearn.requires_dependency.append(python)
    
    spark = DataScienceSoftware("Apache_Spark")
    spark.developed_by.append(apache)
    spark.written_in.extend([scala, python])
    spark.has_license = apache2
    
    pandas = DataScienceSoftware("Pandas")
    pandas.written_in.append(python)
    pandas.supports_format.extend([csv, parquet])
    pandas.interoperates_with.append(sklearn)

    db_platform = CloudPlatform("Databricks_Unified_Analytics")
    db_platform.developed_by.append(databricks_inc)
    db_platform.has_license = proprietary
    db_platform.runs_on_os.append(linux)
    db_platform.requires_dependency.append(spark)
    db_platform.is_deprecated = False    
    jupyter = NotebookEnvironment("JupyterLab")
    jupyter.has_license = mit
    jupyter.interoperates_with.extend([pandas, tf, pt])
    
    ApacheProject.equivalent_to = [
        Software & developed_by.value(apache)
    ]
    
    EnterprisePlatform.equivalent_to = [
        Software & (
            developed_by.value(google) | 
            developed_by.value(microsoft) | 
            developed_by.value(databricks_inc)
        )
    ]

    PurePythonTool.equivalent_to = [
        Software & written_in.value(python)
    ]
    
    software_A = Software("SW_A")
    software_B = Software("SW_B")
    software_C = Software("SW_C")

    
    software_A.implements_algorithm.append(gpt_algo) 
    software_B.written_in.append(python)
    software_C.developed_by.append(apache)
    
    onto.save(file="dsso.owl", format="rdfxml")
    print("\nOntology saved as dsso.owl")
    
    print("\n--- BEFORE REASONING (Results for software A B C specifically) ---")
    print(f"SW A is a: {software_A.is_a}")
    print(f"SW B is a: {software_B.is_a}")
    print(f"SW C is a: {software_C.is_a}")
    
    print("Running Reasoner...")
    sync_reasoner(infer_property_values=True)
    
    print("\n--- INFERENCE RESULTS ---")
    
    if DeepLearningSoftware in pt.is_a:
        print(f"SUCCESS: {pt.name} classified as DeepLearningSoftware")
    else:
        print(f"FAIL: {pt.name} NOT classified correctly")

    if ApacheProject in spark.is_a:
        print(f"SUCCESS: {spark.name} classified as ApacheProject")
    else:
        print(f"FAIL: {spark.name} NOT classified correctly")
    
    if tf in pt.interoperates_with and pt in tf.interoperates_with:
        print(f"SUCCESS: Interoperability is Symmetric between {tf.name} and {pt.name}")
    else:
        print(f"FAIL: Interoperability is NOT Symmetric between {tf.name} and {pt.name}")
            
    print("\n--- AFTER REASONING (Results for software A B C specifically) ---")
    print(f"SW A is now: {software_A.is_a}")
    print(f"SW B is now: {software_B.is_a}")
    print(f"SW C is now: {software_C.is_a}")