{{/*
Expand the name of the chart.
*/}}
{{- define "ml-recommendation-system.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "ml-recommendation-system.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "ml-recommendation-system.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "ml-recommendation-system.labels" -}}
helm.sh/chart: {{ include "ml-recommendation-system.chart" . }}
{{ include "ml-recommendation-system.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "ml-recommendation-system.selectorLabels" -}}
app.kubernetes.io/name: {{ include "ml-recommendation-system.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "ml-recommendation-system.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "ml-recommendation-system.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Common environment variables
*/}}
{{- define "ml-recommendation-system.commonEnv" -}}
- name: ENVIRONMENT
  value: {{ .Values.app.environment | quote }}
- name: DEBUG_MODE
  value: {{ .Values.app.debug | quote }}
- name: REDIS_HOST
  value: {{ include "ml-recommendation-system.fullname" . }}-redis-master
- name: REDIS_PORT
  value: "6379"
- name: KAFKA_BOOTSTRAP_SERVERS
  value: {{ include "ml-recommendation-system.fullname" . }}-kafka:9092
- name: DATABASE_URL
  value: postgresql://{{ .Values.postgresql.auth.username }}:{{ .Values.postgresql.auth.password }}@{{ include "ml-recommendation-system.fullname" . }}-postgresql:5432/{{ .Values.postgresql.auth.database }}
{{- end }}

{{/*
Component labels
*/}}
{{- define "ml-recommendation-system.componentLabels" -}}
{{- $component := . -}}
app.kubernetes.io/component: {{ $component }}
{{- end }}

{{/*
Create image name
*/}}
{{- define "ml-recommendation-system.image" -}}
{{- $registry := .Values.global.imageRegistry | default .Values.image.registry -}}
{{- $repository := .repository -}}
{{- $tag := .tag | default .Chart.AppVersion -}}
{{- printf "%s/%s:%s" $registry $repository $tag -}}
{{- end }}

{{/*
Create Redis connection string
*/}}
{{- define "ml-recommendation-system.redisUrl" -}}
{{- if .Values.redis.enabled }}
redis://{{ include "ml-recommendation-system.fullname" . }}-redis-master:6379
{{- else }}
{{ .Values.externalRedis.url }}
{{- end }}
{{- end }}

{{/*
Create Kafka connection string
*/}}
{{- define "ml-recommendation-system.kafkaServers" -}}
{{- if .Values.kafka.enabled }}
{{ include "ml-recommendation-system.fullname" . }}-kafka:9092
{{- else }}
{{ .Values.externalKafka.servers }}
{{- end }}
{{- end }}

{{/*
Create PostgreSQL connection string
*/}}
{{- define "ml-recommendation-system.postgresqlUrl" -}}
{{- if .Values.postgresql.enabled }}
postgresql://{{ .Values.postgresql.auth.username }}:{{ .Values.postgresql.auth.password }}@{{ include "ml-recommendation-system.fullname" . }}-postgresql:5432/{{ .Values.postgresql.auth.database }}
{{- else }}
{{ .Values.externalPostgresql.url }}
{{- end }}
{{- end }}
