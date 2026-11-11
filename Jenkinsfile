// ==============================================================================
// Jenkinsfile — Multibranch Declarative Pipeline on Master-Slave Architecture
// ==============================================================================
// Runs exclusively on dedicated slave agent node (label: 'mlops-agent').
// Dynamically adjusts pipeline behavior based on active Git branch:
//  - feature/* / PR: Fast Lint, Parallel Model Risk & Quality Gates, API tests
//  - staging: Full Verification, Docker Image Build (:staging), Preview Deploy
//  - main: Full Quality Gates, Docker Build (:latest), Production Vercel Deploy
// ==============================================================================

pipeline {
    agent {
        label 'mlops-agent'  // Targets the dedicated slave/worker node
    }

    environment {
        PYTHON_VERSION      = '3.9'
        VIRTUAL_ENV         = '.venv'
        APP_ENV             = "${env.BRANCH_NAME == 'main' ? 'production' : (env.BRANCH_NAME == 'staging' ? 'staging' : 'development')}"
        DOCKER_IMAGE        = 'rvce-event-intelligence-api'
        VERCEL_TOKEN        = credentials('VERCEL_TOKEN')
        VERCEL_PROJECT_ID   = credentials('VERCEL_PROJECT_ID')
        VERCEL_ORG_ID       = credentials('VERCEL_ORG_ID')
    }

    options {
        timeout(time: 30, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '20'))
        disableConcurrentBuilds()
        ansiColor('xterm')
    }

    stages {
        // ====================================================================
        // STAGE 1: SLAVE AGENT ENVIRONMENT VERIFICATION
        // ====================================================================
        stage('Slave Agent Verification') {
            steps {
                echo "💻 Executing on Slave Worker: ${env.NODE_NAME} (Label: mlops-agent)"
                echo "🌿 Active Git Branch: ${env.BRANCH_NAME}"
                echo "🏷️ Target Environment: ${env.APP_ENV}"
                sh '''
                    echo "--- Slave Host System Info ---"
                    uname -a
                    python3 --version
                    docker --version
                    node --version
                    echo "--- Workspace Directory ---"
                    pwd
                '''
            }
        }

        // ====================================================================
        // STAGE 2: PYTHON VIRTUAL ENVIRONMENT & DEPENDENCIES
        // ====================================================================
        stage('Environment & Dependencies') {
            steps {
                echo "🔧 Preparing Python Virtual Environment on Slave Agent..."
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
        // STAGE 3: STATIC CODE ANALYSIS & LINTING
        // ====================================================================
        stage('Code Quality & Static Analysis') {
            steps {
                echo "🔍 Running Static Code Checks (flake8 & black)..."
                sh '''
                    . ${VIRTUAL_ENV}/bin/activate
                    flake8 app tests --count --select=E9,F63,F7,F82 --show-source --statistics
                    flake8 app tests --count --exit-zero --max-complexity=15 --max-line-length=127 --statistics
                    black --check app tests --line-length=120 || true
                '''
            }
        }

        // ====================================================================
        // STAGE 4: PARALLEL QUALITY GATES (MODEL & API TESTING)
        // ====================================================================
        stage('Automated Quality Gates (Parallel)') {
            parallel {
                stage('Model Performance & Risk Gates') {
                    steps {
                        echo "🧠 Executing Regression, Classification & Model Risk Quality Gates..."
                        sh '''
                            . ${VIRTUAL_ENV}/bin/activate
                            pytest tests/test_model_performance.py -v
                            pytest tests/test_model_risk_quality.py -v
                        '''
                    }
                }
                stage('API & UI SLA Tests') {
                    steps {
                        echo "⚡ Executing FastAPI REST Endpoints, Telemetry & UI SLA Tests..."
                        sh '''
                            . ${VIRTUAL_ENV}/bin/activate
                            pytest tests/test_api.py -v --cov=app --cov-report=term-missing --cov-report=xml:coverage.xml --junitxml=test-results.xml
                        '''
                    }
                }
            }
        }

        // ====================================================================
        // STAGE 5: DOCKER CONTAINERIZATION & SMOKE TEST
        // ====================================================================
        stage('Docker Containerization & Smoke Test') {
            steps {
                echo "🐳 Building Docker Container on Slave Agent..."
                sh '''
                    TAG_NAME="${BRANCH_NAME}-${BUILD_NUMBER}"
                    docker build -t ${DOCKER_IMAGE}:${TAG_NAME} -t ${DOCKER_IMAGE}:${BRANCH_NAME} .
                    
                    # Background smoke container to verify live health probe
                    CONTAINER_NAME="smoke-${BRANCH_NAME}-${BUILD_NUMBER}"
                    docker run -d --name ${CONTAINER_NAME} -p 8002:8000 ${DOCKER_IMAGE}:${TAG_NAME}
                    
                    sleep 6
                    curl --retry 6 --retry-delay 2 --retry-connrefused -f http://localhost:8002/health
                    curl -f http://localhost:8002/ready
                    
                    # Teardown smoke container
                    docker stop ${CONTAINER_NAME} || true
                    docker rm ${CONTAINER_NAME} || true
                '''
            }
        }

        // ====================================================================
        // STAGE 6A: DEPLOY PREVIEW (STAGING & FEATURE BRANCHES)
        // ====================================================================
        stage('Deploy Preview Environment') {
            when {
                anyOf {
                    branch 'staging'
                    branch 'develop'
                    branch 'feature/*'
                    branch 'PR-*'
                }
            }
            steps {
                echo "🌐 Deploying Preview Environment to Vercel for branch ${env.BRANCH_NAME}..."
                sh '''
                    if [ -n "${VERCEL_TOKEN}" ]; then
                        PREVIEW_URL=$(vercel deploy --token=${VERCEL_TOKEN} --yes)
                        echo "🎉 Vercel Preview Deployed: ${PREVIEW_URL}"
                    else
                        echo "ℹ️ VERCEL_TOKEN not set; skipping live cloud preview deploy."
                    fi
                '''
            }
        }

        // ====================================================================
        // STAGE 6B: DEPLOY PRODUCTION (MAIN BRANCH ONLY)
        // ====================================================================
        stage('Deploy Production') {
            when {
                branch 'main'
            }
            steps {
                echo "🚀 Deploying to Vercel Production from branch main..."
                sh '''
                    if [ -n "${VERCEL_TOKEN}" ]; then
                        PROD_URL=$(vercel deploy --prod --token=${VERCEL_TOKEN} --yes)
                        echo "=========================================================="
                        echo "🎉 Production Deployment Live!"
                        echo "🔗 URL: ${PROD_URL}"
                        echo "=========================================================="
                    else
                        echo "ℹ️ VERCEL_TOKEN not set in credentials; using existing live production."
                    fi
                '''
            }
        }
    }

    // ========================================================================
    // POST BUILD AUDIT & NOTIFICATIONS
    // ========================================================================
    post {
        always {
            junit allowEmptyResults: true, testResults: 'test-results.xml'
            cleanWs deleteDirs: true, notFailBuild: true
        }
        success {
            echo "✅ Multibranch Pipeline Run #${env.BUILD_NUMBER} succeeded on branch '${env.BRANCH_NAME}' on agent '${env.NODE_NAME}'!"
        }
        failure {
            echo "❌ Multibranch Pipeline Run #${env.BUILD_NUMBER} failed on branch '${env.BRANCH_NAME}'."
        }
    }
}
