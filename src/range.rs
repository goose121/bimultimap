
pub struct FstRange<T, B: RangeBounds<T>>(B);

impl<T, U, B> RangeBounds<(T, U)> for FstRange<T, B> where
	B: RangeBounds<T>
{
	fn start_bound(&self) -> Bound<&T> {
		
	}
}
