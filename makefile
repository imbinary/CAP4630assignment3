cc = javac
classpath = bin/weka.jar:.

compile:
	$(cc) -cp $(classpath) -d bin/ src/*.java

run-simple:
	java -cp bin/weka.jar:bin SimpleWeka data/simple.arff 10 0.2

run-iris:
	java -cp bin/weka.jar:bin SimpleWeka data/iris.arff 10 0.2

run-labor:
	java -cp bin/weka.jar:bin SimpleWeka data/labor.arff 200 0.15
