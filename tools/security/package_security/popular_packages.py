"""
Comprehensive Popular Package Lists for Multi-Ecosystem Typosquatting Detection.

This module provides curated lists of popular packages across all major
programming language ecosystems to enable effective typosquatting detection.
"""

from typing import Dict, List


class PopularPackageDatabase:
    """Database of popular packages across multiple ecosystems."""
    
    # JavaScript/TypeScript/Node.js Ecosystem
    NPM_POPULAR_PACKAGES = [
        # Core frameworks and libraries
        "react", "vue", "angular", "express", "next", "nuxt", "svelte",
        "lodash", "moment", "axios", "request", "chalk", "commander",
        "inquirer", "yargs", "dotenv", "cors", "helmet", "morgan",
        
        # Build tools and bundlers
        "webpack", "rollup", "parcel", "vite", "esbuild", "babel",
        "typescript", "ts-node", "eslint", "prettier", "jest", "mocha",
        "chai", "cypress", "playwright", "puppeteer",
        
        # State management and utilities
        "redux", "mobx", "zustand", "recoil", "rxjs", "immutable",
        "ramda", "underscore", "date-fns", "dayjs", "uuid", "nanoid",
        
        # Server and API
        "fastify", "koa", "hapi", "socket.io", "ws", "graphql",
        "apollo-server", "prisma", "mongoose", "sequelize", "typeorm",
        
        # Development tools
        "nodemon", "concurrently", "cross-env", "rimraf", "mkdirp",
        "glob", "minimatch", "semver", "validate-npm-package-name",
        
        # Scoped packages (examples)
        "@babel/core", "@babel/preset-env", "@types/node", "@types/react",
        "@angular/core", "@vue/cli", "@nestjs/core", "@storybook/react"
    ]
    
    # Python Ecosystem
    PYTHON_POPULAR_PACKAGES = [
        # Core libraries
        "requests", "urllib3", "setuptools", "pip", "wheel", "six",
        "python-dateutil", "pytz", "certifi", "charset-normalizer",
        
        # Data science and ML
        "numpy", "pandas", "matplotlib", "seaborn", "plotly", "scipy",
        "scikit-learn", "tensorflow", "torch", "pytorch", "keras",
        "xgboost", "lightgbm", "catboost", "statsmodels", "nltk",
        "spacy", "gensim", "opencv-python", "pillow", "imageio",
        
        # Web frameworks
        "django", "flask", "fastapi", "tornado", "pyramid", "bottle",
        "starlette", "uvicorn", "gunicorn", "celery", "redis",
        
        # Database and ORM
        "sqlalchemy", "psycopg2", "pymongo", "peewee", "alembic",
        "sqlite3", "mysql-connector-python", "cx-oracle",
        
        # Testing and development
        "pytest", "unittest2", "nose", "tox", "coverage", "black",
        "flake8", "pylint", "mypy", "isort", "bandit", "safety",
        
        # CLI and utilities
        "click", "argparse", "configparser", "pyyaml", "toml",
        "jsonschema", "validators", "python-dotenv", "environs",
        
        # Async and networking
        "asyncio", "aiohttp", "aiofiles", "httpx", "trio", "anyio"
    ]
    
    # Rust Ecosystem
    RUST_POPULAR_PACKAGES = [
        # Core utilities
        "serde", "serde_json", "serde_derive", "tokio", "async-std",
        "futures", "clap", "structopt", "anyhow", "thiserror", "eyre",
        
        # HTTP and web
        "reqwest", "hyper", "actix-web", "warp", "rocket", "tide",
        "tower", "axum", "tonic", "prost", "serde_urlencoded",
        
        # Data structures and algorithms
        "rand", "uuid", "chrono", "time", "regex", "lazy_static",
        "once_cell", "parking_lot", "crossbeam", "rayon", "itertools",
        
        # Logging and diagnostics
        "log", "env_logger", "tracing", "tracing-subscriber", "slog",
        "flexi_logger", "simplelog", "fern", "pretty_env_logger",
        
        # Database and storage
        "diesel", "sqlx", "rusqlite", "redis", "mongodb", "rocksdb",
        "sled", "redb", "bincode", "postcard", "rmp-serde",
        
        # System and file handling
        "tempfile", "dirs", "walkdir", "notify", "memmap2", "mio",
        "nix", "libc", "winapi", "windows", "core-foundation",
        
        # Crypto and security
        "ring", "rustls", "openssl", "sha2", "md5", "base64", "hex"
    ]
    
    # Java/Kotlin Ecosystem (Maven/Gradle)
    JAVA_POPULAR_PACKAGES = [
        # Core frameworks
        "org.springframework:spring-core", "org.springframework:spring-boot-starter",
        "org.springframework:spring-web", "org.springframework:spring-data-jpa",
        "org.springframework.boot:spring-boot-starter-web",
        "org.springframework.boot:spring-boot-starter-data-jpa",
        
        # Logging
        "org.slf4j:slf4j-api", "ch.qos.logback:logback-classic",
        "org.apache.logging.log4j:log4j-core", "org.apache.logging.log4j:log4j-api",
        
        # Testing
        "org.junit.jupiter:junit-jupiter", "org.junit:junit", "org.mockito:mockito-core",
        "org.assertj:assertj-core", "org.testng:testng", "org.hamcrest:hamcrest",
        
        # Database
        "org.hibernate:hibernate-core", "org.mybatis:mybatis",
        "org.postgresql:postgresql", "mysql:mysql-connector-java",
        "com.h2database:h2", "org.hsqldb:hsqldb",
        
        # HTTP and web
        "com.squareup.okhttp3:okhttp", "org.apache.httpcomponents:httpclient",
        "com.fasterxml.jackson.core:jackson-core", "com.google.code.gson:gson",
        
        # Utilities
        "org.apache.commons:commons-lang3", "commons-io:commons-io",
        "com.google.guava:guava", "org.apache.commons:commons-collections4",
        
        # Build tools
        "org.apache.maven.plugins:maven-compiler-plugin",
        "org.apache.maven.plugins:maven-surefire-plugin"
    ]
    
    # .NET/C# Ecosystem
    DOTNET_POPULAR_PACKAGES = [
        # Core frameworks
        "Microsoft.AspNetCore.App", "Microsoft.NETCore.App",
        "Microsoft.EntityFrameworkCore", "Microsoft.Extensions.Hosting",
        "Microsoft.Extensions.DependencyInjection", "Microsoft.Extensions.Logging",
        
        # Web and API
        "Microsoft.AspNetCore.Mvc", "Microsoft.AspNetCore.Authentication",
        "Microsoft.AspNetCore.Authorization", "Swashbuckle.AspNetCore",
        "Microsoft.AspNetCore.SignalR", "Microsoft.AspNetCore.Cors",
        
        # Data access
        "Microsoft.EntityFrameworkCore.SqlServer", "Dapper", "MongoDB.Driver",
        "StackExchange.Redis", "Npgsql", "MySql.Data", "SQLite",
        
        # Testing
        "Microsoft.NET.Test.Sdk", "xunit", "xunit.runner.visualstudio",
        "NUnit", "MSTest.TestFramework", "Moq", "FluentAssertions",
        
        # JSON and serialization
        "Newtonsoft.Json", "System.Text.Json", "AutoMapper",
        "MessagePack", "protobuf-net", "YamlDotNet",
        
        # HTTP and communication
        "RestSharp", "HttpClientFactory", "Polly", "MediatR",
        "Refit", "Flurl", "Flurl.Http",
        
        # Utilities
        "Serilog", "NLog", "FluentValidation", "BCrypt.Net-Next",
        "HangFire", "Quartz", "Castle.Core"
    ]
    
    # Go Ecosystem
    GO_POPULAR_PACKAGES = [
        # Web frameworks and HTTP
        "github.com/gin-gonic/gin", "github.com/gorilla/mux",
        "github.com/labstack/echo", "github.com/fiber/fiber",
        "github.com/julienschmidt/httprouter", "net/http",
        
        # Database and ORM
        "github.com/jinzhu/gorm", "gorm.io/gorm", "database/sql",
        "github.com/jmoiron/sqlx", "go.mongodb.org/mongo-driver",
        "github.com/go-redis/redis", "github.com/lib/pq",
        
        # CLI and utilities
        "github.com/spf13/cobra", "github.com/spf13/viper",
        "github.com/urfave/cli", "flag", "os", "fmt", "log",
        
        # JSON and serialization
        "encoding/json", "github.com/json-iterator/go",
        "gopkg.in/yaml.v2", "gopkg.in/yaml.v3", "github.com/BurntSushi/toml",
        
        # HTTP client and networking
        "github.com/go-resty/resty", "net/http", "context",
        "github.com/gorilla/websocket", "golang.org/x/net",
        
        # Testing and development
        "testing", "github.com/stretchr/testify", "github.com/onsi/ginkgo",
        "github.com/onsi/gomega", "github.com/golang/mock",
        
        # Concurrency and async
        "sync", "context", "golang.org/x/sync",
        "github.com/panjf2000/ants", "runtime"
    ]
    
    # Ruby Ecosystem
    RUBY_POPULAR_PACKAGES = [
        # Web frameworks
        "rails", "sinatra", "rack", "puma", "unicorn", "thin",
        "roda", "hanami", "grape", "padrino",
        
        # Database and ORM
        "activerecord", "sequel", "datamapper", "mongoid",
        "pg", "mysql2", "sqlite3", "redis", "dalli",
        
        # Testing
        "rspec", "minitest", "test-unit", "cucumber", "factory_bot",
        "faker", "webmock", "vcr", "capybara", "selenium-webdriver",
        
        # Utilities and tools
        "bundler", "rake", "thor", "nokogiri", "json", "httparty",
        "faraday", "rest-client", "mechanize", "mail", "mime-types",
        
        # Background jobs
        "sidekiq", "resque", "delayed_job", "que", "good_job",
        
        # Authentication and security
        "devise", "omniauth", "cancancan", "pundit", "bcrypt",
        
        # Asset pipeline and frontend
        "sprockets", "sass-rails", "coffee-rails", "uglifier",
        "turbo-rails", "stimulus-rails", "webpacker"
    ]
    
    # PHP Ecosystem
    PHP_POPULAR_PACKAGES = [
        # Frameworks
        "laravel/laravel", "symfony/symfony", "codeigniter4/framework",
        "cakephp/cakephp", "zendframework/zendframework", "slim/slim",
        "phalcon/cphalcon", "yiisoft/yii2", "laminas/laminas-mvc",
        
        # Composer tools
        "composer/composer", "phpunit/phpunit", "mockery/mockery",
        "psr/log", "psr/http-message", "psr/container", "psr/cache",
        
        # Database and ORM
        "doctrine/orm", "doctrine/dbal", "illuminate/database",
        "propel/propel", "cycle/orm", "laravel/eloquent",
        
        # HTTP and client libraries
        "guzzlehttp/guzzle", "symfony/http-client", "curl/curl",
        "react/http", "amphp/http-client", "kriswallsmith/buzz",
        
        # Templating
        "twig/twig", "smarty/smarty", "mustache/mustache",
        "league/plates", "symfony/templating",
        
        # Utilities
        "monolog/monolog", "carbon/carbon", "ramsey/uuid",
        "vlucas/phpdotenv", "league/flysystem", "intervention/image",
        "nesbot/carbon", "egulias/email-validator", "swiftmailer/swiftmailer"
    ]
    
    @classmethod
    def get_popular_packages(cls) -> Dict[str, List[str]]:
        """Get comprehensive popular package database for all ecosystems."""
        return {
            # JavaScript/TypeScript/Node.js
            "npm": cls.NPM_POPULAR_PACKAGES,
            "yarn": cls.NPM_POPULAR_PACKAGES,
            "pnpm": cls.NPM_POPULAR_PACKAGES,
            
            # Python
            "pypi": cls.PYTHON_POPULAR_PACKAGES,
            "pip": cls.PYTHON_POPULAR_PACKAGES,
            "poetry": cls.PYTHON_POPULAR_PACKAGES,
            "pipenv": cls.PYTHON_POPULAR_PACKAGES,
            
            # Rust
            "crates.io": cls.RUST_POPULAR_PACKAGES,
            "cargo": cls.RUST_POPULAR_PACKAGES,
            
            # Java/Kotlin
            "maven": cls.JAVA_POPULAR_PACKAGES,
            "gradle": cls.JAVA_POPULAR_PACKAGES,
            "mvn": cls.JAVA_POPULAR_PACKAGES,
            
            # .NET/C#
            "nuget": cls.DOTNET_POPULAR_PACKAGES,
            "dotnet": cls.DOTNET_POPULAR_PACKAGES,
            
            # Go
            "go": cls.GO_POPULAR_PACKAGES,
            "gomod": cls.GO_POPULAR_PACKAGES,
            
            # Ruby
            "gem": cls.RUBY_POPULAR_PACKAGES,
            "rubygems": cls.RUBY_POPULAR_PACKAGES,
            
            # PHP
            "composer": cls.PHP_POPULAR_PACKAGES,
            "packagist": cls.PHP_POPULAR_PACKAGES,
            
            # Others (using sensible defaults)
            "homebrew": ["python", "node", "git", "curl", "wget", "vim", "emacs"],
            "conda": cls.PYTHON_POPULAR_PACKAGES[:20],  # Subset for conda
            "conan": ["boost/1.75.0", "openssl/1.1.1", "zlib/1.2.11", "fmt/7.1.3"]
        }
    
    @classmethod
    def get_ecosystem_aliases(cls) -> Dict[str, str]:
        """Get mapping of ecosystem aliases to canonical names."""
        return {
            "npm": "npm",
            "yarn": "npm", 
            "pnpm": "npm",
            "node": "npm",
            "nodejs": "npm",
            "javascript": "npm",
            "typescript": "npm",
            
            "pip": "pypi",
            "python": "pypi",
            "poetry": "pypi",
            "pipenv": "pypi",
            
            "cargo": "crates.io",
            "rust": "crates.io",
            
            "maven": "maven",
            "gradle": "maven",
            "mvn": "maven",
            "java": "maven",
            "kotlin": "maven",
            
            "nuget": "nuget",
            "dotnet": "nuget",
            "csharp": "nuget",
            "c#": "nuget",
            
            "go": "go",
            "golang": "go",
            "gomod": "go",
            
            "gem": "gem",
            "ruby": "gem",
            "rubygems": "gem",
            
            "composer": "composer",
            "php": "composer",
            "packagist": "composer"
        }