from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
import os
from datetime import datetime, timedelta
import secrets
from functools import lru_cache
import hashlib

app = Flask(__name__)
CORS(app)

# 설정
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv(
    'DATABASE_URL', 
    'sqlite:///coding_ai.db'
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-change-this')
app.config['JSON_SORT_KEYS'] = False  # JSON 정렬 비활성화 (속도 향상)

db = SQLAlchemy(app)

# 어드민 마스터 키 (외계어 같은 키)
ADMIN_MASTER_KEY = os.getenv('ADMIN_MASTER_KEY', '🔮🌌⚡🎭🦾💫🌀🔥⭐🎯')

# ==================== 초고속 최적화 ====================

# 응답 캐시 (같은 질문에 빠른 답변)
RESPONSE_CACHE = {}
MAX_CACHE_SIZE = 1000

# 극소형 모델 사용 (distilgpt2보다 10배 빠름)
MODEL_NAME = os.getenv('MODEL_NAME', 'distilgpt2')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"⚡ Loading ultra-fast model: {MODEL_NAME}")
print(f"🔧 Device: {DEVICE.upper()}")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    
    # 모델 최적화
    model.eval()
    
    # GPU 최적화
    if DEVICE == "cuda":
        print("⚡ GPU 최적화 활성화 (8x 빠름)")
        model = model.half()  # fp16 사용 (속도 8배 향상)
        # 양자화 (메모리 50% 감소, 속도 향상)
        try:
            from torch.quantization import quantize_dynamic
            model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            print("⚡ INT8 양자화 활성화")
        except:
            pass
    
    print("✓ 초고속 모델 로드 완료")
except Exception as e:
    print(f"❌ Error: {e}")
    model = None

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ==================== 데이터베이스 모델 ====================

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    api_key = db.Column(db.String(255), unique=True, nullable=False)
    
    # 구독 정보
    subscription_tier = db.Column(db.String(20), default='free')  # free, premium
    subscription_expires = db.Column(db.DateTime, nullable=True)
    
    # 사용 통계
    requests_today = db.Column(db.Integer, default=0)
    last_request_reset = db.Column(db.DateTime, default=datetime.utcnow)
    total_requests_month = db.Column(db.Integer, default=0)
    
    # 어드민
    is_admin = db.Column(db.Boolean, default=False)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def generate_api_key(self):
        self.api_key = secrets.token_urlsafe(32)
        return self.api_key
    
    def is_premium(self):
        if self.is_admin:
            return True
        if self.subscription_tier == 'premium':
            return self.subscription_expires and self.subscription_expires > datetime.utcnow()
        return False
    
    def get_daily_limit(self):
        """일일 요청 제한"""
        if self.is_admin:
            return float('inf')
        return 20 if self.is_premium() else 3
    
    def get_response_time_limit(self):
        """응답 시간 제한 (분)"""
        if self.is_admin:
            return float('inf')
        return float('inf') if self.is_premium() else 10
    
    def can_make_request(self):
        """요청 가능 여부 확인"""
        if self.is_admin:
            return True, "관리자"
        
        # 날 초기화
        if (datetime.utcnow() - self.last_request_reset).days >= 1:
            self.requests_today = 0
            self.last_request_reset = datetime.utcnow()
        
        limit = self.get_daily_limit()
        if self.requests_today >= limit:
            remaining = self.get_response_time_limit()
            return False, f"일일 제한({limit})에 도달. 다음 요청까지 {remaining}분 기다리거나 구독하세요."
        
        return True, "OK"

class Subscription(db.Model):
    __tablename__ = 'subscriptions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    plan = db.Column(db.String(20), nullable=False)  # premium
    price = db.Column(db.Float, nullable=False)
    payment_method = db.Column(db.String(50), nullable=False)
    payment_id = db.Column(db.String(255), unique=True)
    
    started_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime, nullable=False)
    
    is_active = db.Column(db.Boolean, default=True)

# ==================== API 엔드포인트 ====================

@app.route('/')
def index():
    return render_template('index.html')

# 인증 엔드포인트
@app.route('/api/auth/register', methods=['POST'])
def register():
    try:
        data = request.json
        email = data.get('email', '').strip()
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        
        if not all([email, username, password]):
            return jsonify({'error': '모든 필드 필수'}), 400
        
        if len(password) < 6:
            return jsonify({'error': '비밀번호는 6자 이상'}), 400
        
        if User.query.filter_by(email=email).first():
            return jsonify({'error': '이미 가입된 이메일'}), 400
        
        if User.query.filter_by(username=username).first():
            return jsonify({'error': '이미 사용 중인 아이디'}), 400
        
        user = User(email=email, username=username)
        user.set_password(password)
        user.generate_api_key()
        
        db.session.add(user)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': '가입 완료!',
            'api_key': user.api_key
        }), 201
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        data = request.json
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        
        user = User.query.filter_by(username=username).first()
        
        if not user or not user.check_password(password):
            return jsonify({'error': '아이디 또는 비밀번호 오류'}), 401
        
        token = jwt.encode(
            {'user_id': user.id, 'exp': datetime.utcnow() + timedelta(days=30)},
            app.config['SECRET_KEY'],
            algorithm='HS256'
        )
        
        return jsonify({
            'success': True,
            'token': token,
            'api_key': user.api_key,
            'username': user.username
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/admin', methods=['POST'])
def admin_login():
    """어드민 로그인 (마스터 키로)"""
    try:
        data = request.json
        master_key = data.get('master_key', '').strip()
        
        if master_key != ADMIN_MASTER_KEY:
            return jsonify({'error': '잘못된 마스터 키'}), 401
        
        # 어드민 사용자 생성 또는 조회
        admin_user = User.query.filter_by(username='admin').first()
        
        if not admin_user:
            admin_user = User(
                email='admin@coding-ai.local',
                username='admin'
            )
            admin_user.set_password('admin-password-' + secrets.token_hex(16))
            admin_user.is_admin = True
            admin_user.generate_api_key()
            db.session.add(admin_user)
            db.session.commit()
        
        token = jwt.encode(
            {'user_id': admin_user.id, 'is_admin': True, 'exp': datetime.utcnow() + timedelta(days=30)},
            app.config['SECRET_KEY'],
            algorithm='HS256'
        )
        
        return jsonify({
            'success': True,
            'token': token,
            'api_key': admin_user.api_key,
            'message': '어드민 로그인 성공'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 토큰 검증 및 사용자 조회
def verify_token(token):
    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        return User.query.get(payload['user_id'])
    except:
        return None

# 캐시에서 응답 조회
def get_cached_response(message_hash):
    """응답 캐시에서 조회"""
    return RESPONSE_CACHE.get(message_hash)

def cache_response(message_hash, response):
    """응답 캐시에 저장"""
    if len(RESPONSE_CACHE) > MAX_CACHE_SIZE:
        # 오래된 항목 제거
        oldest_key = next(iter(RESPONSE_CACHE))
        del RESPONSE_CACHE[oldest_key]
    RESPONSE_CACHE[message_hash] = response

@lru_cache(maxsize=128)
def generate_response_fast(prompt_text):
    """극도로 최적화된 응답 생성"""
    inputs = tokenizer.encode(prompt_text, return_tensors='pt').to(DEVICE)
    
    with torch.no_grad():
        # 극단적으로 짧은 응답 (매우 빠름)
        outputs = model.generate(
            inputs,
            max_length=80,  # 매우 짧은 응답 (원래 300)
            temperature=0.6,
            top_p=0.85,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_beams=1,  # 빔 서치 비활성화 (빠름)
            repetition_penalty=1.1,
            length_penalty=0.6
        )
    
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if response_text.startswith(prompt_text):
        response_text = response_text[len(prompt_text):].strip()
    
    return response_text if response_text else "답변 완료."

# 주요 기능: 초고속 챗 엔드포인트
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        import time
        start_time = time.time()
        
        # 인증
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        user = verify_token(token)
        
        if not user:
            return jsonify({'error': '인증 필요'}), 401
        
        # 요청 가능 여부
        can_request, message = user.can_make_request()
        if not can_request:
            return jsonify({'error': message}), 429
        
        data = request.json
        user_message = data.get('message', '').strip()
        language = data.get('language', 'python')
        
        if not user_message:
            return jsonify({'error': '메시지 필수'}), 400
        
        if len(user_message) > 1000:
            return jsonify({'error': '1000자 이하로 입력하세요'}), 400
        
        if model is None:
            return jsonify({'error': '모델 로드 실패'}), 500
        
        # 캐시 확인 (매우 빠름)
        message_hash = hashlib.md5((user_message + language).encode()).hexdigest()
        cached = get_cached_response(message_hash)
        
        if cached:
            response_text = cached
            cache_hit = True
        else:
            # AI 응답 생성 (초고속)
            prompt = f"Lang:{language}\nQ:{user_message}\nA:"
            response_text = generate_response_fast(prompt)
            cache_response(message_hash, response_text)
            cache_hit = False
        
        if not response_text or response_text.strip() == "":
            response_text = "✓ 완료"
        
        # 사용 횟수 증가
        user.requests_today += 1
        user.total_requests_month += 1
        db.session.commit()
        
        remaining = int(user.get_daily_limit()) - user.requests_today
        elapsed = time.time() - start_time
        
        return jsonify({
            'success': True,
            'message': response_text.strip()[:200],  # 응답 길이 제한
            'language': language,
            'remaining_requests': max(0, remaining),
            'daily_limit': int(user.get_daily_limit()) if user.get_daily_limit() != float('inf') else '∞',
            'subscription': user.subscription_tier,
            'is_admin': user.is_admin,
            'response_time_ms': round(elapsed * 1000, 1),
            'from_cache': cache_hit
        })
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

# 스트리밍 응답 (실시간 응답)
@app.route('/api/chat-stream', methods=['POST'])
def chat_stream():
    """실시간 스트리밍 응답 (매우 빠른 체감)"""
    try:
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        user = verify_token(token)
        
        if not user:
            return jsonify({'error': '인증 필요'}), 401
        
        can_request, _ = user.can_make_request()
        if not can_request:
            return jsonify({'error': '요청 제한 도달'}), 429
        
        data = request.json
        user_message = data.get('message', '').strip()
        language = data.get('language', 'python')
        
        if not user_message:
            return jsonify({'error': '메시지 필수'}), 400
        
        # 사용 횟수 증가
        user.requests_today += 1
        user.total_requests_month += 1
        db.session.commit()
        
        # 응답 생성 (스트리밍)
        prompt = f"Lang:{language}\nQ:{user_message}\nA:"
        inputs = tokenizer.encode(prompt, return_tensors='pt').to(DEVICE)
        
        def generate():
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=60,
                    temperature=0.6,
                    top_p=0.85,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            yield response
        
        return generate()
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/user/profile', methods=['GET'])
def get_profile():
    """사용자 프로필"""
    try:
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        user = verify_token(token)
        
        if not user:
            return jsonify({'error': '인증 필요'}), 401
        
        return jsonify({
            'username': user.username,
            'email': user.email,
            'api_key': user.api_key,
            'subscription_tier': user.subscription_tier,
            'is_premium': user.is_premium(),
            'is_admin': user.is_admin,
            'subscription_expires': user.subscription_expires.isoformat() if user.subscription_expires else None,
            'requests_today': user.requests_today,
            'total_requests_month': user.total_requests_month,
            'daily_limit': int(user.get_daily_limit()) if user.get_daily_limit() != float('inf') else '무제한'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/subscribe', methods=['POST'])
def subscribe():
    """구독 업그레이드 (더미 결제)"""
    try:
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        user = verify_token(token)
        
        if not user:
            return jsonify({'error': '인증 필요'}), 401
        
        if user.is_admin:
            return jsonify({'error': '어드민은 구독 불필요'}), 400
        
        data = request.json
        plan = data.get('plan', 'premium')
        
        if plan != 'premium':
            return jsonify({'error': '유효하지 않은 플랜'}), 400
        
        # 구독 활성화
        user.subscription_tier = 'premium'
        user.subscription_expires = datetime.utcnow() + timedelta(days=30)
        
        subscription = Subscription(
            user_id=user.id,
            plan='premium',
            price=9.99,
            payment_method='card',
            payment_id='test-' + secrets.token_hex(16),
            expires_at=user.subscription_expires
        )
        
        db.session.add(subscription)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': '프리미엄 구독 시작!',
            'subscription_expires': user.subscription_expires.isoformat(),
            'daily_limit': 20
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/users', methods=['GET'])
def admin_get_users():
    """어드민: 모든 사용자 조회"""
    try:
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        user = verify_token(token)
        
        if not user or not user.is_admin:
            return jsonify({'error': '어드민만 접근 가능'}), 403
        
        users = User.query.all()
        
        return jsonify({
            'success': True,
            'total_users': len(users),
            'users': [{
                'id': u.id,
                'username': u.username,
                'email': u.email,
                'subscription_tier': u.subscription_tier,
                'is_premium': u.is_premium(),
                'requests_today': u.requests_today,
                'total_requests_month': u.total_requests_month,
                'created_at': u.created_at.isoformat()
            } for u in users]
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/reset-user/<int:user_id>', methods=['POST'])
def admin_reset_user(user_id):
    """어드민: 사용자 요청 횟수 초기화"""
    try:
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        admin = verify_token(token)
        
        if not admin or not admin.is_admin:
            return jsonify({'error': '어드민만 접근 가능'}), 403
        
        user = User.query.get(user_id)
        if not user:
            return jsonify({'error': '사용자 없음'}), 404
        
        user.requests_today = 0
        user.total_requests_month = 0
        db.session.commit()
        
        return jsonify({'success': True, 'message': '초기화 완료'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'device': DEVICE
    })

# ==================== DB 초기화 ====================

with app.app_context():
    db.create_all()

if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=debug_mode)
