resource "aws_s3_bucket" "animaldet" {
  bucket = var.project_name

  tags = merge(
    var.tags,
    {
      Name = var.project_name
    }
  )
}

resource "aws_s3_bucket_versioning" "animaldet" {
  bucket = aws_s3_bucket.animaldet.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "animaldet" {
  bucket = aws_s3_bucket.animaldet.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "animaldet" {
  bucket = aws_s3_bucket.animaldet.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_policy" "animaldet" {
  bucket = aws_s3_bucket.animaldet.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "DenyInsecureTransport"
        Effect    = "Deny"
        Principal = "*"
        Action    = "s3:*"
        Resource = [
          aws_s3_bucket.animaldet.arn,
          "${aws_s3_bucket.animaldet.arn}/*"
        ]
        Condition = {
          Bool = {
            "aws:SecureTransport" = "false"
          }
        }
      }
    ]
  })
}

variable "tags" {
  description = "Common tags for all resources"
  type        = map(string)
  default     = {}
}

variable "project_name" {
  description = "Project name"
  type        = string
}

output "bucket_name" {
  description = "Name of the S3 bucket"
  value       = aws_s3_bucket.animaldet.id
}

output "bucket_arn" {
  description = "ARN of the S3 bucket"
  value       = aws_s3_bucket.animaldet.arn
}