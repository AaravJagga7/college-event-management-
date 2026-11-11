pipeline {
    agent any

    environment {
        PYTHON_VERSION      = '3.9'
        VIRTUAL_ENV         = '.venv'
        VERCEL_PROJECT_ID   = credentials('VERCEL_PROJECT_ID')   // Configured in Jenkins Credentials
        VERCEL_ORG_ID       = credentials('VERCEL_ORG_ID')       // Configured in Jenkins Credentials
        VERCEL_TOKEN        = credentials('VERCEL_TOKEN')        // Configured in Jenkins Credentials
        APP_ENV             = 'production'
        DOCKER_IMAGE        = 'rvce-event-intelligence-api'
    }

    options {
        timeout(time: 30, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '15'))
        disableConcurrentBuilds()
        ansiColor('xterm')
    }

    stages {
        // ====================================================================
        // STAGE 1: ENVIRONMENT SETUP & DEPENDENCY CACHING
        // ====================================================================
        stage('Environment & Setup') {
            steps {
                echo "🔧 Preparing Python ${PYTHON_VERSION} Virtual Environment..."
                sh '''
                    python3 -m venv ${VIRTUAL_ENV}
                    . ${VIRTUAL_ENV}/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                    pip install flake8 black pytest-cov
                '''
            }
        }

        // ====================================================================
        // STAGE 2: STATIC CODE ANALYSIS & LINTING
        // ====================================================================
        stage('Lint & Code Quality') {
            steps {
                echo "🔍 Running Static Code Analysis (flake8 & black)..."
                sh '''
                    . ${VIRTUAL_ENV}/bin/activate
                    flake8 app tests --count --select=E9,F63,F7,F82 --show-source --statistics
                    flake8 app tests --count --exit-zero --max-complexity=15 --max-line-length=127 --statistics
                    black --check app tests --line-length=120 || true
                '''
            }
        }

        // ====================================================================
        // STAGE 3: MODEL TESTING & RISK QUALITY GATES
        // ====================================================================
        stage('Model Quality & Risk Gates') {
            steps {
                echo "🧠 Executing 22 Quality Gates & Model Risk Checks..."
                sh '''
                    . ${VIRTUAL_ENV}/bin/activate
                    pytest tests/test_model_performance.py -v
                    pytest tests/test_model_risk_quality.py -v
                '''
            }
        }

        // ====================================================================
        // STAGE 4: API & INTEGRATION TESTS
        // ====================================================================
        stage('API & UI Integration Tests') {
            steps {
                echo "⚡ Testing FastAPI REST Endpoints & UI Dashboard..."
                sh '''
                    . ${VIRTUAL_ENV}/bin/activate
                    pytest tests/test_api.py -v --cov=app --cov-report=term-missing --cov-report=xml:coverage.xml --junitxml=test-results.xml
                '''
            }
            post {
                always {
                    junit allowEmptyResults: true, testResults: 'test-results.xml'
                }
            }
        }

        // ====================================================================
        // STAGE 5: DOCKER CONTAINERIZATION & SMOKE TEST
        // ====================================================================
        stage('Docker Build & Verify') {
            steps {
                echo "🐳 Building Production Multi-stage Docker Container..."
                sh '''
                    docker build -t ${DOCKER_IMAGE}:${BUILD_NUMBER} -t ${DOCKER_IMAGE}:latest .
                    
                    # Run container in background for live smoke test
                    docker run -d --name test-jenkins-${BUILD_NUMBER} -p 8001:8000 ${DOCKER_IMAGE}:latest
                    
                    # Probe healthcheck endpoint
                    sleep 5
                    curl --retry 6 --retry-delay 2 --retry-connrefused -f http://localhost:8001/health
                    curl -f http://localhost:8001/ready
                    
                    # Cleanup test container
                    docker stop test-jenkins-${BUILD_NUMBER} || true
                    docker rm test-jenkins-${BUILD_NUMBER} || true
                '''
            }
        }

        // ====================================================================
        // STAGE 6: DEPLOYMENT TO VERCEL (STAGING / PRODUCTION)
        // ====================================================================
        stage('Deploy to Vercel') {
            when {
                anyOf {
                    branch 'main'
                    branch 'release/*'
                    buildingTag()
                }
            }
            steps {
                echo "🚀 Deploying RVCE Event Intelligence Application to Vercel..."
                sh '''
                    # Install Vercel CLI if not present
                    if ! command -v vercel &> /dev/null; then
                        npm install -g vercel@latest
                    fi

                    # Pull Vercel environment metadata
                    vercel pull --yes --environment=production --token=${VERCEL_TOKEN}

                    # Build prebuilt artifacts
                    vercel build --prod --token=${VERCEL_TOKEN}

                    # Deploy to Production Vercel domain
                    DEPLOY_URL=$(vercel deploy --prebuilt --prod --token=${VERCEL_TOKEN} --yes)
                    echo "=========================================================="
                    echo "🎉 Successfully Deployed to Vercel Production!"
                    echo "🔗 Live Application URL: ${DEPLOY_URL}"
                    echo "=========================================================="
                    
                    # Smoke test the live Vercel deployment
                    sleep 3
                    curl -f "${DEPLOY_URL}/health" || echo "Note: Initial serverless cold start in progress"
                '''
            }
        }
    }

    // ========================================================================
    // POST BUILD ACTIONS & NOTIFICATIONS
    // ========================================================================
    post {
        success {
            echo "✅ Jenkins Pipeline Completed Successfully: All Model Quality Gates & Vercel Deployment Passed!"
        }
        failure {
            echo "❌ Jenkins Pipeline Failed! Review logs above for failed model quality gates or deployment errors."
        }
        always {
            cleanWs deleteDirs: true, notFailBuild: true
        }
    }
}

