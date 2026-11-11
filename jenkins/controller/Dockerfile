# ==============================================================================
# Jenkins Controller (Master) Dockerfile
# ==============================================================================
# Orchestrates multibranch pipelines, hosts the web dashboard, manages
# credentials, and delegates all build execution to dedicated slave agents.
# ==============================================================================
FROM jenkins/jenkins:lts-jdk17

USER root

# Install plugins using jenkins-plugin-cli
COPY plugins.txt /usr/share/jenkins/ref/plugins.txt
RUN jenkins-plugin-cli --plugin-file /usr/share/jenkins/ref/plugins.txt --verbose

# Pre-create directory for Jenkins Configuration as Code (JCasC)
RUN mkdir -p /var/jenkins_home/casc_configs /var/jenkins_home/secrets && \
    chown -R jenkins:jenkins /var/jenkins_home

USER jenkins

EXPOSE 8080 50000
