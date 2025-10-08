#!/bin/bash
# 多Agent加密货币量化交易系统 - 启动脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_message() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}"
    echo "=============================================="
    echo "  多Agent加密货币量化交易系统"
    echo "  Multi-Agent Crypto Trading System"
    echo "=============================================="
    echo -e "${NC}"
}

# 显示帮助信息
show_help() {
    cat << EOF
多Agent加密货币量化交易系统启动脚本

用法: $0 [选项]

选项:
    start           启动所有服务
    stop            停止所有服务
    restart         重启所有服务
    status          查看服务状态
    logs            查看服务日志
    setup           设置开发环境
    test            运行测试
    analyze         运行性能分析
    help            显示此帮助信息

示例:
    $0 start        # 启动服务
    $0 test         # 运行测试
    $0 analyze      # 性能分析
    $0 setup        # 初始设置

EOF
}

# 检查依赖
check_dependencies() {
    print_message "检查系统依赖..."

    # 检查Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker 未安装，请先安装Docker"
        exit 1
    fi

    # 检查Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose 未安装，请先安装Docker Compose"
        exit 1
    fi

    # 检查Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 未安装，请先安装Python3"
        exit 1
    fi

    print_message "依赖检查完成 ✓"
}

# 设置开发环境
setup_environment() {
    print_message "设置开发环境..."

    # 创建.env文件（如果不存在）
    if [ ! -f .env ]; then
        if [ -f scripts/deployment/.env.example ]; then
            cp scripts/deployment/.env.example .env
            print_message "已创建 .env 文件，请根据需要修改配置"
        fi
    fi

    # 安装Python依赖
    if [ -f scripts/deployment/requirements.txt ]; then
        pip3 install -r scripts/deployment/requirements.txt
        print_message "Python依赖安装完成 ✓"
    fi

    print_message "环境设置完成！"
}

# 启动服务
start_services() {
    print_message "启动系统服务..."

    if [ -f scripts/deployment/docker-compose.yml ]; then
        cd scripts/deployment
        docker-compose up -d
        cd ../..
        print_message "服务启动完成 ✓"

        # 显示服务状态
        print_message "服务状态:"
        docker-compose -f scripts/deployment/docker-compose.yml ps
    else
        print_error "docker-compose.yml 文件不存在"
        exit 1
    fi
}

# 停止服务
stop_services() {
    print_message "停止系统服务..."

    if [ -f scripts/deployment/docker-compose.yml ]; then
        cd scripts/deployment
        docker-compose down
        cd ../..
        print_message "服务停止完成 ✓"
    else
        print_warning "docker-compose.yml 文件不存在"
    fi
}

# 重启服务
restart_services() {
    print_message "重启系统服务..."
    stop_services
    sleep 2
    start_services
}

# 查看服务状态
show_status() {
    print_message "查看服务状态..."

    if [ -f scripts/deployment/docker-compose.yml ]; then
        docker-compose -f scripts/deployment/docker-compose.yml ps
    else
        print_warning "docker-compose.yml 文件不存在"
    fi
}

# 查看日志
show_logs() {
    print_message "查看服务日志..."

    if [ -f scripts/deployment/docker-compose.yml ]; then
        docker-compose -f scripts/deployment/docker-compose.yml logs -f
    else
        print_warning "docker-compose.yml 文件不存在"
    fi
}

# 运行测试
run_tests() {
    print_message "运行测试套件..."

    if [ -f tools/analysis/run_tests.py ]; then
        python3 tools/analysis/run_tests.py
    else
        print_warning "测试脚本不存在"
    fi
}

# 运行性能分析
run_analysis() {
    print_message "运行性能分析..."

    if [ -f project_manager.py ]; then
        python3 project_manager.py analyze
    else
        print_warning "项目管理工具不存在"
    fi
}

# 主函数
main() {
    print_header

    case "${1:-help}" in
        "start")
            check_dependencies
            start_services
            ;;
        "stop")
            stop_services
            ;;
        "restart")
            restart_services
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs
            ;;
        "setup")
            check_dependencies
            setup_environment
            ;;
        "test")
            run_tests
            ;;
        "analyze")
            run_analysis
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_error "未知命令: $1"
            show_help
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"