pipeline {
    agent any

    environment {
        IMAGE_NAME = "parthivreddy2005/ml-api:latest"
        CONTAINER_NAME = "2022bcs0073_jenkins"
        PORT = "8000"
    }

    stages {

        stage('Pull Image') {
            steps {
                script {
                    sh "docker pull ${IMAGE_NAME}"
                }
            }
        }

        stage('Run Container') {
            steps {
               script {
                   sh """
                   docker rm -f ${CONTAINER_NAME} || true
                   docker run -d -p ${PORT}:8000 --name ${CONTAINER_NAME} ${IMAGE_NAME}
                   """
               }
            }
        }

        stage('Wait for Service Readiness') {
            steps {
                script {
                    sleep 10
                    sh """
                    curl -f http://host.docker.internal:${PORT}/docs
                    """
                }
            }
        }

        stage('Send Valid Inference Request') {
            steps {
                script {
                    def response = sh(
                        script: "curl -s -X POST http://host.docker.internal:${PORT}/predict -H 'Content-Type: application/json' -d @tests/valid.json",
                        returnStdout: true
                    ).trim()

                    echo "Valid Response: ${response}"

                    if (!response.contains("predicted_wine_quality")) {
                        error("Prediction field missing!")
                    }
                    echo "Valid response structure verified"
                }
            }
        }

        stage('Send Invalid Request') {
            steps {
                script {
                    def response = sh(
                        script: "curl -s -X POST http://host.docker.internal:${PORT}/predict -H 'Content-Type: application/json' -d @tests/invalid.json",
                        returnStdout: true
                    ).trim()

                    echo "Invalid Response: ${response}"

                    if (!response.contains("detail")) {
                        error("Invalid input did not return validation error!")
                    }
                    echo "Invalid input correctly returned validation error"
                }
            }
        }

        stage('Stop Container') {
            steps {
                script {
                    sh "docker stop ${CONTAINER_NAME}"
                    sh "docker rm ${CONTAINER_NAME}"
                }
            }
        }
    }
}
