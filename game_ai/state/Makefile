# java Makefile

PROJ_NAME = $(shell basename $(CURDIR))

#---------------------------------------------------------------------------------------------------------
# Main App
MAIN = ai.ch2.StateDemo

# Packages 
PACKAGES = ai.ch2

# Libraries
#LIB_PATH = /home/SANCLEMENTE/fran/lib/java/
#LIB_PATH = /home/zeroth/work/lib/java/
#LIB_CP = lib/ds.jar
LIB_PATH = 
LIB_CP = 

# Source path
SPATH = src/main/java/
SRCS = $(foreach PKG, $(PACKAGES), $(wildcard $(SPATH)$(subst .,/,$(PKG)))/*.java)

# Destiny path
DPATH = target/classes/

# Classpath
CP = .:$(DPATH):$(LIB_CP)

#---------------------------------------------------------------------------------------------------------
# Distribution
JAR_PATH = target/dist/

#---------------------------------------------------------------------------------------------------------
# Documentation
DOC_PATH = doc/
JDOC_PATH = jdoc/

#---------------------------------------------------------------------------------------------------------
# Tests
TEST_CLASS = ai.ch2.StateTest 
TEST_RUNNER = org.junit.runner.JUnitCore

# Packages 
TEST_PACKAGES = ai.ch2

# Libraries
JUNIT4_CP = $(LIB_PATH)junit4/junit-4.13.1.jar:$(LIB_PATH)junit4/hamcrest-core-1.3.jar
TEST_LIB_CP = $(JUNIT4_CP)

# Source path
TEST_SPATH = src/test/java/
TEST_SRCS = $(foreach PKG, $(TEST_PACKAGES), $(wildcard $(TEST_SPATH)$(subst .,/,$(PKG)))/*.java)

# Destiny path
TEST_DPATH = target/test-classes/

# Test classpath
TEST_CP = .:$(DPATH):$(TEST_DPATH):$(TEST_LIB_CP):$(LIB_CP)

#---------------------------------------------------------------------------------------------------------
# Targets

JVM = java
JFLAGS = -g
JC = javac
JAR = jar
JDOC = javadoc
MDIR = mkdir -p

classes: 
	$(JC) -d $(DPATH) -cp $(CP) $(SRCS)

classes-debug: 
	$(JC) $(JFLAGS) -d $(DPATH) -cp $(CP) $(SRCS)

run: classes
	$(JVM) -cp $(CP) $(MAIN)

run-jar: jar
	$(JVM) -cp $(CP) -jar $(JAR_PATH)$(PROJ_NAME).jar

debug: classes-debug
	$(JVM) -cp $(CP) -Xdebug -Xrunjdwp:transport=dt_socket,address=8000,server=y,suspend=n $(MAIN)

jar: classes
	$(RM) -r $(JAR_PATH)*
	$(MDIR) -p $(JAR_PATH)
	$(JAR) cvfe $(JAR_PATH)$(PROJ_NAME).jar $(MAIN) -C $(DPATH) .

test-classes: classes
	$(JC) -cp $(TEST_CP) -d $(TEST_DPATH) $(TEST_SRCS)

test: test-classes
	$(JVM) -cp $(TEST_CP) $(TEST_RUNNER) $(TEST_CLASS)

jdocs:
	$(JDOC) -cp $(CP) --frames -d $(JDOC_PATH) -sourcepath $(SPATH) $(PACKAGES)

clean: 
	$(RM) -r $(DPATH)*
	$(RM) -r $(TEST_DPATH)*
	$(RM) -r $(JAR_PATH)*
	$(RM) -r $(JDOC_PATH)*
	$(RM) -r $(JAR_PATH)*

init:
	#$(RM) -r *  # Dangerous!!!
	$(MDIR) -p $(foreach PKG_DIR, $(PACKAGES), $(SPATH)$(subst .,/,$(PKG_DIR)))
	$(MDIR) -p $(foreach PKG_DIR, $(TEST_PACKAGES), $(TEST_SPATH)$(subst .,/,$(PKG_DIR)))

