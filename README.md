# WellWiseAI: Oil & Gas Data Ingestion Pipeline

A comprehensive data ingestion pipeline for oil and gas well data, designed to parse various file formats and store structured data in Astra DB with vector search capabilities for unstructured documents.

## 🚀 Overview

WellWiseAI is a robust data processing system that ingests oil and gas well data from multiple file formats, converts them to a canonical schema, and stores them in Astra DB. The system supports both structured data (tables) and unstructured documents (vector collections) with advanced search capabilities.

## 📁 Project Structure

```
WellWiseAI/
├── parsers/                 # File format parsers
│   ├── __init__.py
│   ├── las.py              # LAS well log parser
│   ├── dlis.py             # DLIS well log parser
│   ├── csv_parser.py       # CSV drilling data parser
│   ├── dat.py              # Well picks parser (with LLM enhancement)
│   ├── xlsx.py             # XLSX facies interpretation parser (with LLM enhancement)
│   ├── survey.py           # Well survey/trajectory parser (with LLM enhancement)
│   ├── segy.py             # SEGY seismic data parser
│   ├── lti.py              # LTI (Log Tape Image) parser
│   └── unstructured.py     # Unstructured document parser (with LLM enhancement)
├── db/                     # Database utilities
│   ├── insert_data.py      # Astra DB insertion module
│   ├── vector_store.py     # Vector collection management
│   ├── clean_table.py      # Table cleanup utility
│   └── create_table.py     # Table creation utility
├── schema/                 # Data schema definitions
│   ├── canonical.json      # Canonical schema (75 fields)
│   ├── fields.py           # Field definitions
│   └── table.cql           # CQL table definition
├── wellwise_parser.py     # Main parsing orchestrator
├── wellwise_pipeline.py   # End-to-end pipeline
├── requirements.txt       # Python dependencies
└── .env-example            # Environment configuration example
```

## 🔧 Features

### 📊 Structured Data Parsing
- **LAS Files**: Well log data with curve measurements
- **DLIS Files**: Digital Log Interchange Standard files
- **CSV Files**: Drilling and operational data
- **DAT Files**: Well picks and geological markers (with LLM enhancement)
- **XLSX Files**: Facies interpretation data (with LLM enhancement)
- **Survey Files**: Well trajectory and survey data (with LLM enhancement)
- **SEGY Files**: Seismic data with intelligent sampling
- **LTI Files**: Log Tape Image files (Schlumberger SFINX format)

### 📄 Unstructured Document Processing
- **PDF Files**: Technical reports, well reports, geological studies
- **XLSX Files**: Spreadsheets and tabular documents
- **TXT Files**: Text documents and reports
- **Vector Search**: Semantic search capabilities with Astra DB

### 🤖 AI/LLM Integration
- **OpenAI GPT-4o-mini**: Enhanced metadata extraction
- **Geological Interpretation**: Automated facies classification
- **Context Enhancement**: Intelligent field mapping and validation

### 🗄️ Database Integration
- **Astra DB**: Cloud-native Cassandra database
- **Structured Tables**: `petro_data` table with 75 canonical fields
- **Vector Collections**: `oil_gas_documents` for semantic search
- **Hybrid Search**: Combined keyword and vector search

## 📋 Canonical Schema

The system uses a standardized 75-field canonical schema covering:

### Core Fields
- **Well Identification**: `well_id`, `record_type`, `curve_name`
- **Depth Measurements**: `depth_start`, `depth_end`, `sample_interval`
- **Statistical Analysis**: `sample_mean`, `sample_min`, `sample_max`, `sample_stddev`

### Geological Data
- **Petrophysical Properties**: `porosity`, `water_saturation`, `permeability`, `vshale`
- **Lithology Flags**: `carbonate_flag`, `coal_flag`, `sand_flag`
- **Formation Data**: `horizon_name`, `horizon_depth`, `facies_code`

### Seismic Data
- **Seismic Attributes**: `seismic_inline`, `seismic_xline`, `seismic_trace_count`
- **Sample Statistics**: `num_samples`, `seismic_sample_rate`

### Metadata
- **Processing Info**: `processing_date`, `processing_software`, `tool_type`
- **Quality Control**: `qc_flag`, `analyst`, `null_count`
- **Geographic Data**: `latitude`, `longitude`, `country`, `field_name`

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd WellWiseAI

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration
```

### 2. Environment Configuration

```bash
# Astra DB Configuration
ASTRA_DB_TOKEN=your_token_here
ASTRA_DB_API_ENDPOINT=your_endpoint_here
ASTRA_DB_KEYSPACE=your_keyspace_here
ASTRA_TABLE_NAME=petro_data

# OpenAI Configuration (for LLM enhancement)
OPENAI_API_KEY=your_openai_key_here
CHAT_COMPLETION_MODEL=gpt-4o-mini

# Data Directories
DATA_DIRECTORY=/path/to/your/data
PARSED_DIRECTORY=structured_data
UNSTRUCTURED_DIRECTORY=parsed_data

# Parser Configuration
SEGY_SAMPLE_SIZE=10000  # Number of traces to sample from SEGY files
```

### 3. Run the Pipeline

```bash
# Full pipeline (parsing + insertion)
python wellwise_pipeline.py

# Parse only (skip database insertion)
python wellwise_pipeline.py --skip-insertion

# Insert only (skip parsing)
python wellwise_pipeline.py --skip-parsing
```

## 📊 Parser Details

### Structured Parsers

#### LAS Parser (`parsers/las.py`)
- **Purpose**: Parse LAS (Log ASCII Standard) well log files
- **Output**: Well log curves and measurements
- **LLM Enhancement**: ❌ None
- **Key Features**: Curve extraction, depth mapping, unit conversion

#### DLIS Parser (`parsers/dlis.py`)
- **Purpose**: Parse DLIS (Digital Log Interchange Standard) files
- **Output**: Digital well log data with frame information
- **LLM Enhancement**: ❌ None
- **Key Features**: Frame-based parsing, curve extraction

#### CSV Parser (`parsers/csv_parser.py`)
- **Purpose**: Parse CSV drilling and operational data
- **Output**: Drilling parameters and operational metrics
- **LLM Enhancement**: ❌ None
- **Key Features**: Column mapping, data validation

#### DAT Parser (`parsers/dat.py`)
- **Purpose**: Parse well picks and geological markers
- **Output**: Formation tops and geological boundaries
- **LLM Enhancement**: ✅ Active (limited to first 5 records)
- **Key Features**: UTM to WGS84 conversion, geological interpretation

#### XLSX Parser (`parsers/xlsx.py`)
- **Purpose**: Parse facies interpretation data
- **Output**: Lithology and facies classifications
- **LLM Enhancement**: ✅ Active
- **Key Features**: Geological context enhancement, facies confidence

#### Survey Parser (`parsers/survey.py`)
- **Purpose**: Parse well survey and trajectory data
- **Output**: Wellbore geometry and directional data
- **LLM Enhancement**: ✅ Active
- **Key Features**: Trajectory calculation, azimuth/inclination processing

#### SEGY Parser (`parsers/segy.py`)
- **Purpose**: Parse seismic data files
- **Output**: Seismic traces and attributes
- **LLM Enhancement**: ❌ Disabled
- **Key Features**: Intelligent sampling, trace statistics, frequency analysis
- **Configuration**: `SEGY_SAMPLE_SIZE` environment variable

#### LTI Parser (`parsers/lti.py`)
- **Purpose**: Parse Log Tape Image files (Schlumberger SFINX)
- **Output**: CPI (Computer Processed Interpretation) data
- **LLM Enhancement**: ❌ None
- **Key Features**: Binary format parsing, realistic data generation

### Unstructured Parser

#### Unstructured Parser (`parsers/unstructured.py`)
- **Purpose**: Parse PDF, XLSX, TXT documents
- **Output**: Contextual documents for vector search
- **LLM Enhancement**: ✅ Active
- **Key Features**: Document chunking, metadata enhancement, semantic search

## 🔍 Search Capabilities

### Structured Data Search
- **Primary Key Queries**: By well_id, record_type, curve_name
- **Range Queries**: Depth-based filtering
- **Indexed Fields**: facies_code, gas_oil_ratio, production_rate, etc.

### Vector Search
- **Semantic Search**: Find similar documents
- **Hybrid Search**: Combine keyword and vector search
- **Metadata Filtering**: Filter by document type, source, etc.

## ⚙️ Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ASTRA_DB_TOKEN` | Astra DB application token | Required |
| `ASTRA_DB_API_ENDPOINT` | Astra DB API endpoint | Required |
| `ASTRA_DB_KEYSPACE` | Database keyspace | `wellwise` |
| `ASTRA_TABLE_NAME` | Table name | `petro_data` |
| `OPENAI_API_KEY` | OpenAI API key for LLM | Optional |
| `CHAT_COMPLETION_MODEL` | LLM model | `gpt-4o-mini` |
| `DATA_DIRECTORY` | Source data directory | Required |
| `PARSED_DIRECTORY` | Structured output directory | `structured_data` |
| `UNSTRUCTURED_DIRECTORY` | Unstructured output directory | `parsed_data` |
| `SEGY_SAMPLE_SIZE` | SEGY trace sampling size | `1000` |

### File Type Configuration

Configure which file types are processed as structured vs unstructured in `.env`:

```bash
STRUCTURED_FILE_TYPES=.las,.dlis,.csv,.dat,.xlsx,.segy,.lti
UNSTRUCTURED_FILE_TYPES=.pdf,.xlsx,.txt
DUAL_PROCESSING_FILE_TYPES=.xlsx  # Process as both structured and unstructured
```

## 🚀 Performance Features

### Parallel Processing
- **ThreadPoolExecutor**: Parallel file processing
- **Configurable Workers**: Set via environment variables
- **Batch Insertion**: Database insertion in batches

### Memory Optimization
- **Streaming Parsing**: Large files processed in chunks
- **Intelligent Sampling**: SEGY files sampled instead of full processing
- **Garbage Collection**: Automatic memory management

### Error Handling
- **Graceful Degradation**: Continue processing on individual file failures
- **Detailed Logging**: Comprehensive error tracking
- **Data Validation**: Schema compliance checking

## 📈 Data Quality

### Validation Rules
- **Primary Key Validation**: Ensures required fields are present
- **Type Conversion**: Automatic data type conversion for database compatibility
- **Range Validation**: Field value range checking
- **Null Handling**: Proper null value management

### Quality Control
- **QC Flags**: Quality control indicators
- **Data Completeness**: Missing value tracking
- **Consistency Checks**: Cross-field validation

## 🔧 Utilities

### Database Management
```bash
# Create database table
python db/create_table.py

# Clean database table
python db/clean_table.py

# Test data insertion
python db/insert_data.py
```

### Parser Testing
```bash
# Test individual parsers
python -c "from parsers.las import LasParser; parser = LasParser('file.las'); result = parser.parse()"
```

## 📊 Monitoring and Logging

### Log Files
- **`wellwise_parser.log`**: Main parsing operations

### Metrics
- **Processing Time**: Per file and total pipeline time
- **Success/Failure Rates**: File processing statistics
- **Data Volume**: Records processed and inserted
- **LLM Usage**: API calls and enhancement statistics

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Ensure all parsers pass validation
5. Update documentation
6. Submit pull request

### Adding New Parsers
1. Create parser class in `parsers/` directory
2. Implement `parse()` method returning canonical format
3. Add to `parser_map` in `wellwise_parser.py`
4. Update file type configuration
5. Add tests and documentation

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

- **Astra DB**: Cloud-native database platform
- **OpenAI**: LLM enhancement capabilities
- **Oil & Gas Industry**: Domain expertise and use cases

---

**WellWiseAI** - Transforming oil and gas data into actionable insights through intelligent parsing and AI-powered enhancement. 